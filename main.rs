// Cargo.toml
// [dev-dependencies]
// itertools = "0.13"

use itertools::Itertools;
use petgraph::algo::toposort;
use petgraph::graph::DiGraph;
use petgraph::prelude::NodeIndex;
use std::collections::{HashMap, HashSet, VecDeque};

/* =========================
 * Global atoms (Factors) with Union-Find + Split Registry
 * ========================= */

type FactorId = u32;
type TensorId = usize;
type OpId = usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisDecomp(pub Vec<FactorId>); // axis = product of factors in order

#[derive(Clone, Debug)]
pub struct Factor {
    pub id: FactorId,
    pub size: u64,
    pub parent: Option<FactorId>,
    pub children: Vec<FactorId>, // order matters
    pub is_concat: bool,
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Vec<u64>,         // exterior: axis 0 is leftmost
    pub decomp: Vec<AxisDecomp>, // canonical, normalized to leaf reps
}

#[derive(Default, Debug)]
pub struct State {
    next_factor: FactorId,
    pub factors: HashMap<FactorId, Factor>,
    pub uf_parent: HashMap<FactorId, FactorId>, // union-find parent
    pub tensors: HashMap<TensorId, Tensor>,
    pub next_tensor_id: TensorId,
    // split cache: (parent_rep, parts) -> children reps
    split_cache: HashMap<(FactorId, Vec<u64>), Vec<FactorId>>,
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc_tensor_id(&mut self) -> TensorId {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        id
    }

    pub fn add_source_tensor(&mut self, shape: Vec<u64>) -> TensorId {
        let id = self.alloc_tensor_id();
        let decomp = shape
            .iter()
            .map(|&len| AxisDecomp(vec![self.new_factor(len, None, false)]))
            .collect();
        let mut t = Tensor { id, shape, decomp };
        self.normalize_tensor(&mut t); // canonicalize to leaf reps
        self.tensors.insert(id, t);
        id
    }

    fn new_factor(&mut self, size: u64, parent: Option<FactorId>, is_concat: bool) -> FactorId {
        let id = self.next_factor;
        self.next_factor += 1;
        self.factors.insert(
            id,
            Factor {
                id,
                size,
                parent,
                children: vec![],
                is_concat,
            },
        );
        self.uf_parent.insert(id, id);
        id
    }

    /* ---------- Union-Find ---------- */
    fn uf_find(&mut self, x: FactorId) -> FactorId {
        let p = *self.uf_parent.get(&x).unwrap_or(&x);
        if p == x {
            x
        } else {
            let r = self.uf_find(p);
            self.uf_parent.insert(x, r);
            r
        }
    }
    fn uf_union(&mut self, a: FactorId, b: FactorId) -> bool {
        let mut ra = self.uf_find(a);
        let mut rb = self.uf_find(b);
        if ra == rb {
            return false;
        }
        // size must match unless concat
        let fa = &self.factors[&ra];
        let fb = &self.factors[&rb];
        assert!(
            fa.is_concat == fb.is_concat,
            "cannot alias concat with non-concat"
        );
        assert_eq!(fa.size, fb.size, "alias size mismatch");
        // pick smaller id as root (stable)
        if rb < ra {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.uf_parent.insert(rb, ra);
        true
    }

    /// Read-only find: 경로 압축 없이 대표만 추적
    pub fn uf_find_ro(&self, x: FactorId) -> FactorId {
        let mut r = x;
        loop {
            let p = *self.uf_parent.get(&r).unwrap_or(&r);
            if p == r {
                return r;
            }
            r = p;
        }
    }

    /// 읽기 전용: rep 기준으로 leaf factor들까지 펼쳐 out에 추가
    pub fn flatten_to_leaf_reps_ro(&self, f: FactorId, out: &mut Vec<FactorId>) {
        let r = self.uf_find_ro(f);
        let node = &self.factors[&r];
        if node.children.is_empty() {
            out.push(r);
        } else {
            for &c in &node.children {
                self.flatten_to_leaf_reps_ro(c, out);
            }
        }
    }

    /// 읽기 전용: AxisDecomp를 leaf rep들의 선형 리스트로 정규화
    pub fn normalize_axis_vec_ro(&self, v: &AxisDecomp) -> Vec<FactorId> {
        let mut out = vec![];
        for &f in &v.0 {
            self.flatten_to_leaf_reps_ro(f, &mut out);
        }
        out
    }

    /* ---------- Normalization to leaf reps ---------- */
    fn normalize_axis_vec(&mut self, v: &AxisDecomp) -> Vec<FactorId> {
        let mut out = vec![];
        for &f in &v.0 {
            let r = self.uf_find(f);
            self.flatten_to_leaf_reps(r, &mut out);
        }
        out
    }
    fn flatten_to_leaf_reps(&mut self, f: FactorId, out: &mut Vec<FactorId>) {
        let r = self.uf_find(f);
        let node = self.factors[&r].clone();
        if node.children.is_empty() {
            out.push(r);
        } else {
            for c in node.children {
                let r = self.uf_find(c);
                self.flatten_to_leaf_reps(r, out);
            }
        }
    }
    fn normalize_tensor(&mut self, t: &mut Tensor) {
        let mut new_decomp = Vec::with_capacity(t.decomp.len());
        for ax in 0..t.decomp.len() {
            let v = self.normalize_axis_vec(&t.decomp[ax]);
            new_decomp.push(AxisDecomp(v));
        }
        t.decomp = new_decomp;
    }

    /* ---------- Split with cache (on representative) ---------- */
    fn split_factor(&mut self, parent: FactorId, parts: &[u64]) -> Vec<FactorId> {
        let prep = self.uf_find(parent);
        let key = (prep, parts.to_vec());
        if let Some(v) = self.split_cache.get(&key) {
            return v.clone();
        }
        let p = self.factors[&prep].size;
        let q: u64 = parts.iter().product();
        assert_eq!(
            p, q,
            "split parts must multiply to parent size (p={p}, q={q})"
        );
        let kids: Vec<FactorId> = parts
            .iter()
            .map(|&sz| self.new_factor(sz, Some(prep), false))
            .collect();
        self.factors.get_mut(&prep).unwrap().children = kids.clone();
        // all children are own reps
        self.split_cache.insert(key, kids.clone());
        kids
    }

    fn new_concat_axis(&mut self, len: u64) -> AxisDecomp {
        AxisDecomp(vec![self.new_factor(len, None, true)])
    }

    fn new_tensor_from_view(&mut self, shape: Vec<u64>, decomp: Vec<AxisDecomp>) -> TensorId {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let mut t = Tensor { id, shape, decomp };
        self.normalize_tensor(&mut t);
        self.tensors.insert(id, t);
        id
    }
}

/* =========================
 * Graph model
 * ========================= */

#[derive(Clone, Debug)]
pub enum OpKind {
    // single input, single output
    Split { axis: usize, parts: Vec<u64> }, // interior: split one axis into parts
    Merge { groups: Vec<Vec<usize>> },      // interior: merge axes per groups order
    Permute { perm: Vec<usize> },           // interior: reorder axes
    // multi-input, single output
    Concat { axis: usize }, // interior: concat along axis (inputs must match other axes)
    Einsum { spec: String }, // interior: "ab,bc->ac" (N inputs)
}

#[derive(Clone, Debug)]
pub struct Operator {
    pub id: OpId,
    pub kind: OpKind,           // interior semantics
    pub inputs: Vec<TensorId>,  // exterior tensor ids (axis 0 is leftmost)
    pub outputs: Vec<TensorId>, // exterior tensor ids (preallocated or empty to be filled)
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub operators: Vec<Operator>,
    pub tensor_producer: HashMap<TensorId, Option<OpId>>,
    pub tensor_consumers: HashMap<TensorId, Vec<OpId>>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_operator(
        &mut self,
        kind: OpKind,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
    ) -> OpId {
        let id = self.operators.len();
        self.operators.push(Operator {
            id,
            kind,
            inputs: inputs.clone(),
            outputs: outputs.clone(),
        });

        // outputs: 반드시 단 한 연산자만 producer
        for &t in &outputs {
            if let Some(Some(prev)) = self.tensor_producer.get(&t) {
                panic!("Tensor #{t} already has a producer: op{prev}");
            }
            self.tensor_producer.insert(t, Some(id));
        }
        // inputs: consumer 등록
        for &t in &inputs {
            self.tensor_consumers.entry(t).or_default().push(id);
            self.tensor_producer.entry(t).or_insert(None); // source로 표시
        }
        id
    }

    pub fn consumers_of(&self, tid: TensorId) -> impl Iterator<Item = OpId> + '_ {
        self.tensor_consumers
            .get(&tid)
            .into_iter()
            .flatten()
            .copied()
    }

    pub fn topo_order(&self) -> Vec<OpId> {
        // 1) petgraph로 오퍼레이터 DAG 구성
        let mut pg: DiGraph<OpId, ()> = DiGraph::new();
        // op id -> node index 매핑 테이블
        let mut op_nodes: Vec<NodeIndex> = Vec::with_capacity(self.operators.len());
        for op in &self.operators {
            op_nodes.push(pg.add_node(op.id));
        }

        // 2) 텐서의 producer → consumers 간 에지 추가
        //    (producer가 없는 source 텐서는 에지 없음)
        for (tid, prod_opt) in &self.tensor_producer {
            if let Some(prod) = prod_opt {
                let p = *prod;
                if let Some(cons) = self.tensor_consumers.get(tid) {
                    for &c in cons {
                        pg.add_edge(op_nodes[p], op_nodes[c], ());
                    }
                }
            }
        }

        // 3) 위상 정렬 (사이클 존재 시 패닉/에러)
        let order = toposort(&pg, None).expect("Cycle detected in operator graph");
        // NodeIndex → OpId
        order.into_iter().map(|ni| pg[ni]).collect()
    }
}

/* =========================
 * Interior views & helpers
 * ========================= */

#[derive(Clone, Debug)]
struct View {
    shape: Vec<u64>,
    decomp: Vec<AxisDecomp>, // NOTE: not normalized; normalize on-demand
}

fn view_from_tensor(st: &mut State, tid: TensorId) -> View {
    let t = &st.tensors[&tid];
    View {
        shape: t.shape.clone(),
        decomp: t.decomp.clone(),
    }
}

fn normalize_view(st: &mut State, v: &View) -> View {
    let decomp = v
        .decomp
        .iter()
        .map(|ax| AxisDecomp(st.normalize_axis_vec(ax)))
        .collect();
    View {
        shape: v.shape.clone(),
        decomp,
    }
}

/* ---------- Primitive interior transforms ---------- */

fn apply_split(st: &mut State, vin: &View, axis: usize, parts: &[u64]) -> View {
    let v = normalize_view(st, vin);
    // require a single factor on that axis
    let AxisDecomp(fs0) = &v.decomp[axis];
    assert_eq!(fs0.len(), 1, "Split assumes single-factor axis");
    let kids = st.split_factor(fs0[0], parts);
    let mut shape = Vec::with_capacity(v.shape.len() - 1 + parts.len());
    shape.extend_from_slice(&v.shape[..axis]);
    shape.extend_from_slice(parts);
    shape.extend_from_slice(&v.shape[axis + 1..]);
    let mut decomp = Vec::with_capacity(shape.len());
    decomp.extend_from_slice(&v.decomp[..axis]);
    for &kid in &kids {
        decomp.push(AxisDecomp(vec![kid]));
    }
    decomp.extend_from_slice(&v.decomp[axis + 1..]);
    View { shape, decomp }
}

fn apply_merge(st: &mut State, vin: &View, groups: &[Vec<usize>]) -> View {
    let v = normalize_view(st, vin);
    let shape = groups
        .iter()
        .map(|g| g.iter().map(|&ax| v.shape[ax]).product())
        .collect::<Vec<_>>();
    let decomp = groups
        .iter()
        .map(|g| {
            let mut vec = vec![];
            for &ax in g {
                vec.extend(v.decomp[ax].0.iter().copied());
            }
            AxisDecomp(vec)
        })
        .collect();
    View { shape, decomp }
}

fn apply_permute(st: &mut State, vin: &View, perm: &[usize]) -> View {
    let v = normalize_view(st, vin);
    assert_eq!(perm.len(), v.shape.len());
    let shape = perm.iter().map(|&i| v.shape[i]).collect();
    let decomp = perm.iter().map(|&i| v.decomp[i].clone()).collect();
    View { shape, decomp }
}

fn apply_concat_with_unify(st: &mut State, vins: &mut [View], axis: usize) -> (View, bool) {
    assert!(!vins.is_empty());
    for v in vins.iter_mut() {
        *v = normalize_view(st, v);
    }
    let rank = vins[0].shape.len();
    for v in &*vins {
        assert_eq!(v.shape.len(), rank, "rank mismatch for concat");
    }
    // unify non-concat axes across all inputs to first
    let mut changed = false;
    for ax in 0..rank {
        if ax == axis {
            continue;
        }
        let base = vins[0].decomp[ax].clone();
        for i in 1..vins.len() {
            if unify_axis_decompose(st, &base, &mut vins[i].decomp[ax]) {
                changed = true;
            }
        }
    }
    // build output
    let mut shape = vins[0].shape.clone();
    shape[axis] = vins.iter().map(|v| v.shape[axis]).sum();
    let mut decomp = vins[0].decomp.clone();
    decomp[axis] = st.new_concat_axis(shape[axis]);
    (View { shape, decomp }, changed)
}

/* =========================
 * Einsum kernel (with unify/expand)
 * ========================= */

fn parse_einsum_spec(spec: &str) -> (Vec<Vec<char>>, Vec<char>) {
    // "ab,bc->ac"
    let (lhs, rhs) = spec
        .split_once("->")
        .expect("einsum spec must contain '->'");
    let in_labels: Vec<Vec<char>> = lhs.split(',').map(|s| s.chars().collect()).collect();
    let out_labels: Vec<char> = rhs.chars().collect();
    (in_labels, out_labels)
}

/// Unify two axis decompositions by aliasing equal-size factors and splitting larger ones.
/// Returns true if any union/split happened.
fn unify_axis_decompose(st: &mut State, left: &AxisDecomp, right: &mut AxisDecomp) -> bool {
    let mut changed = false;
    // work on local lists that we mutate as we split
    let mut l = st.normalize_axis_vec(left);
    let mut r = st.normalize_axis_vec(right);
    let mut i = 0usize;
    let mut j = 0usize;
    while i < l.len() && j < r.len() {
        let la = st.uf_find(l[i]);
        let lb = st.uf_find(r[j]);
        let sa = st.factors[&la].size;
        let sb = st.factors[&lb].size;
        if sa == sb {
            // alias reps (may be no-op)
            if st.uf_union(la, lb) {
                changed = true;
            }
            i += 1;
            j += 1;
        } else if sa > sb {
            // split la into [sb, sa/sb]
            assert!(
                sa % sb == 0,
                "cannot unify: sizes not divisible ({} vs {})",
                sa,
                sb
            );
            let kids = st.split_factor(la, &[sb, sa / sb]);
            // replace l[i] with kids
            l.splice(i..=i, kids);
            changed = true;
        } else {
            // sb > sa
            assert!(
                sb % sa == 0,
                "cannot unify: sizes not divisible ({} vs {})",
                sb,
                sa
            );
            let kids = st.split_factor(lb, &[sa, sb / sa]);
            r.splice(j..=j, kids);
            changed = true;
        }
    }
    assert!(
        i == l.len() && j == r.len(),
        "cannot unify: total factor product mismatch"
    );

    // write-back right normalized (left is base, but no need to write)
    right.0 = r;
    changed
}

// 1) 입력 랭크를 라벨 수에 맞게 "자동 Merge"하는 헬퍼 추가
fn coerce_view_rank_for_labels(st: &mut State, vin: &View, labs: &[char]) -> View {
    // 규칙: R = vin.rank, L = labs.len()
    // - R == L: 그대로 사용
    // - R >  L: 앞쪽 축들을 묶어 L개의 축이 되도록 "연속 병합"
    //           (표준형: 첫 그룹에 R-L+1개, 나머지는 1개씩)
    // - R <  L: 현재 구현에선 지원 안 함(명시적 split 필요)
    let R = vin.shape.len();
    let L = labs.len();
    if R == L {
        return normalize_view(st, vin);
    }
    assert!(
        R >= L,
        "einsum: input rank {} < labels {} 는 자동 split이 필요합니다(현재 미지원).",
        R,
        L
    );
    // R > L 인 경우: 연속 병합 그룹 생성
    let first_group_len = R - L + 1;
    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(L);
    let mut idx = 0usize;

    // 그룹0: 앞에서 first_group_len개 축을 전부 병합
    let mut g0 = Vec::with_capacity(first_group_len);
    for _ in 0..first_group_len {
        g0.push(idx);
        idx += 1;
    }
    groups.push(g0);

    // 나머지 그룹: 각 축 하나씩
    while idx < R {
        groups.push(vec![idx]);
        idx += 1;
    }

    let vin = normalize_view(st, vin);
    apply_merge(st, &vin, &groups)
}

fn apply_einsum_with_unify(
    st: &mut State,
    vins: &mut [View],
    spec: &str,
) -> (View, EinsumRelabelMeta, bool) {
    let (in_labels, out_labels) = parse_einsum_spec(spec);
    assert_eq!(in_labels.len(), vins.len(), "inputs vs spec mismatch");

    // ✨ 각 입력을 라벨 개수에 맞게 "암시적 merge"로 맞춰줌
    for (v, labs) in vins.iter_mut().zip(in_labels.iter()) {
        let coerced = coerce_view_rank_for_labels(st, v, labs);
        *v = coerced;
    }

    // 이후는 동일: 정규화 + 라벨별 unify + 출력 구성
    for v in vins.iter_mut() {
        *v = normalize_view(st, v);
    }

    // --- 기존 로직 그대로 ---
    let mut changed = false;
    let mut label_first: HashMap<char, (usize, usize)> = HashMap::new();
    for (i, labs) in in_labels.iter().enumerate() {
        assert_eq!(
            vins[i].shape.len(),
            labs.len(),
            "internal error: coerce failed"
        );
        for (ax, &lab) in labs.iter().enumerate() {
            if let Some(&(fi, fax)) = label_first.get(&lab) {
                let base = vins[fi].decomp[fax].clone();
                if unify_axis_decompose(st, &base, &mut vins[i].decomp[ax]) {
                    changed = true;
                }
            } else {
                label_first.insert(lab, (i, ax));
            }
        }
    }

    let mut out_shape = Vec::with_capacity(out_labels.len());
    let mut out_decomp = Vec::with_capacity(out_labels.len());
    let out_set: HashSet<char> = out_labels.iter().copied().collect();

    for &lab in &out_labels {
        let (i, ax) = label_first[&lab];
        out_decomp.push(vins[i].decomp[ax].clone());
        out_shape.push(vins[i].shape[ax]);
    }

    let mut contracted_axes: Vec<(usize, usize)> = vec![];
    for (i, labs) in in_labels.iter().enumerate() {
        for (ax, &lab) in labs.iter().enumerate() {
            if !out_set.contains(&lab) {
                contracted_axes.push((i, ax));
            }
        }
    }

    (
        View {
            shape: out_shape,
            decomp: out_decomp,
        },
        EinsumRelabelMeta {
            in_labels,
            out_labels,
            contracted_axes,
        },
        changed,
    )
}

/* =========================
 * Relabel mapping (per-op)
 * ========================= */

#[derive(Clone, Debug)]
pub struct OpRelabel {
    pub op: OpId,
    // factor id -> operator-local label (0..)
    pub factor_label_of: HashMap<FactorId, usize>,
    // per output axis: local labels (in-order, left→right)
    pub output_axis_labels: Vec<Vec<usize>>,
    // einsum only: contracted axes labels in (input idx, axis idx, labels)
    pub contracted_axis_labels: Vec<(usize, usize, Vec<usize>)>,
}

fn build_op_relabel_for_simple(st: &mut State, opid: OpId, out_view: &View) -> OpRelabel {
    let mut lbl: HashMap<FactorId, usize> = HashMap::new();
    let mut next = 0usize;

    let mut out_axis_labels = vec![];
    for AxisDecomp(fs) in &normalize_view(st, out_view).decomp {
        let mut axis_lbls = vec![];
        for &f in fs {
            let id = *lbl.entry(f).or_insert_with(|| {
                let cur = next;
                next += 1;
                cur
            });
            axis_lbls.push(id);
        }
        out_axis_labels.push(axis_lbls);
    }

    OpRelabel {
        op: opid,
        factor_label_of: lbl,
        output_axis_labels: out_axis_labels,
        contracted_axis_labels: vec![],
    }
}

#[derive(Clone, Debug)]
struct EinsumRelabelMeta {
    in_labels: Vec<Vec<char>>,
    out_labels: Vec<char>,
    contracted_axes: Vec<(usize, usize)>, // (input idx, axis idx)
}

fn build_op_relabel_for_einsum(
    st: &mut State,
    opid: OpId,
    out_view: &View,
    inputs: &[View],
    meta: &EinsumRelabelMeta,
) -> OpRelabel {
    let mut lbl: HashMap<FactorId, usize> = HashMap::new();
    let mut next = 0usize;

    // outputs first (L→R)
    let mut out_axis_labels = vec![];
    for AxisDecomp(fs) in &normalize_view(st, out_view).decomp {
        let mut axis_lbls = vec![];
        for &f in fs {
            let id = *lbl.entry(f).or_insert_with(|| {
                let cur = next;
                next += 1;
                cur
            });
            axis_lbls.push(id);
        }
        out_axis_labels.push(axis_lbls);
    }

    // contracted axes: by input idx, then axis L→R
    let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, ax) in &meta.contracted_axes {
        grouped.entry(*i).or_default().push(*ax);
    }
    let mut contracted_axis_labels = vec![];
    for i in grouped.keys().copied().sorted() {
        let mut axes = grouped[&i].clone();
        axes.sort_unstable();
        for ax in axes {
            let AxisDecomp(fs) = &normalize_view(st, &inputs[i]).decomp[ax];
            let mut labels = vec![];
            for &f in fs {
                let id = *lbl.entry(f).or_insert_with(|| {
                    let cur = next;
                    next += 1;
                    cur
                });
                labels.push(id);
            }
            contracted_axis_labels.push((i, ax, labels));
        }
    }

    OpRelabel {
        op: opid,
        factor_label_of: lbl,
        output_axis_labels: out_axis_labels,
        contracted_axis_labels,
    }
}

/* =========================
 * Worklist relabel (Method 1 with fixpoint)
 * ========================= */

pub fn relabel_graph(st: &mut State, g: &Graph) -> HashMap<OpId, OpRelabel> {
    let topo = g.topo_order();
    let mut q: VecDeque<OpId> = topo.into();
    let mut inq: HashSet<OpId> = q.iter().copied().collect();

    // We'll update tensors' outputs canonically and enqueue consumers when changed.
    while let Some(opid) = q.pop_front() {
        inq.remove(&opid);
        let op = &g.operators[opid];

        println!("op: {:?}", op);

        match &op.kind {
            OpKind::Split { axis, parts } => {
                let mut vin = view_from_tensor(st, op.inputs[0]);
                let vout = apply_split(st, &vin, *axis, parts);
                let mut t = Tensor {
                    id: op.outputs[0],
                    shape: vout.shape.clone(),
                    decomp: vout.decomp.clone(),
                };
                st.normalize_tensor(&mut t);
                if update_tensor_if_changed(st, t) {
                    for c in g.consumers_of(op.outputs[0]) {
                        if inq.insert(c) {
                            q.push_back(c);
                        }
                    }
                }
            }
            OpKind::Merge { groups } => {
                let vin = view_from_tensor(st, op.inputs[0]);
                let vout = apply_merge(st, &vin, groups);
                let mut t = Tensor {
                    id: op.outputs[0],
                    shape: vout.shape.clone(),
                    decomp: vout.decomp.clone(),
                };
                st.normalize_tensor(&mut t);
                if update_tensor_if_changed(st, t) {
                    for c in g.consumers_of(op.outputs[0]) {
                        if inq.insert(c) {
                            q.push_back(c);
                        }
                    }
                }
            }
            OpKind::Permute { perm } => {
                let vin = view_from_tensor(st, op.inputs[0]);
                let vout = apply_permute(st, &vin, perm);
                let mut t = Tensor {
                    id: op.outputs[0],
                    shape: vout.shape.clone(),
                    decomp: vout.decomp.clone(),
                };
                st.normalize_tensor(&mut t);
                if update_tensor_if_changed(st, t) {
                    for c in g.consumers_of(op.outputs[0]) {
                        if inq.insert(c) {
                            q.push_back(c);
                        }
                    }
                }
            }
            OpKind::Concat { axis } => {
                let mut vins: Vec<View> =
                    op.inputs.iter().map(|&t| view_from_tensor(st, t)).collect();
                let (vout, changed) = apply_concat_with_unify(st, &mut vins, *axis);
                let mut t = Tensor {
                    id: op.outputs[0],
                    shape: vout.shape.clone(),
                    decomp: vout.decomp.clone(),
                };
                st.normalize_tensor(&mut t);
                let out_changed = update_tensor_if_changed(st, t);
                if changed {
                    // alias/split happened on inputs: enqueue their consumers as well
                    let wrote = writeback_normalized_inputs(st, op);
                    // 입력이 갱신되었으면 해당 소비자들을 다시 enqueue
                    if wrote {
                        for &tin in &op.inputs {
                            for c in g.consumers_of(tin) {
                                if inq.insert(c) {
                                    q.push_back(c);
                                }
                            }
                        }
                    }
                }
                if out_changed {
                    for c in g.consumers_of(op.outputs[0]) {
                        if inq.insert(c) {
                            q.push_back(c);
                        }
                    }
                }
            }
            OpKind::Einsum { spec } => {
                let mut vins: Vec<View> =
                    op.inputs.iter().map(|&t| view_from_tensor(st, t)).collect();
                let (vout, _meta, changed) = apply_einsum_with_unify(st, &mut vins, spec);
                let mut t = Tensor {
                    id: op.outputs[0],
                    shape: vout.shape.clone(),
                    decomp: vout.decomp.clone(),
                };
                st.normalize_tensor(&mut t);
                let out_changed = update_tensor_if_changed(st, t);
                if changed {
                    // input structure refined → enqueue their consumers
                    let wrote = writeback_normalized_inputs(st, op);
                    if wrote {
                        for &tin in &op.inputs {
                            for c in g.consumers_of(tin) {
                                if inq.insert(c) {
                                    q.push_back(c);
                                }
                            }
                        }
                    }
                }
                if out_changed {
                    for c in g.consumers_of(op.outputs[0]) {
                        if inq.insert(c) {
                            q.push_back(c);
                        }
                    }
                }
            }
        }
    }

    println!("st: {:#?}", st);

    // // Build per-op relabel report from the final state (non-mutating)
    // let mut reports: HashMap<OpId, OpRelabel> = HashMap::new();
    // for op in &g.operators {
    //     println!("op: {:?}", op);
    //     match &op.kind {
    //         OpKind::Einsum { spec } => {
    //             // recompute views without mutating (should already be unified at fixpoint)
    //             let mut vins: Vec<View> =
    //                 op.inputs.iter().map(|&t| view_from_tensor(st, t)).collect();
    //             let (vout, meta, _changed) = apply_einsum_with_unify(st, &mut vins, spec);
    //             let rep = build_op_relabel_for_einsum(st, op.id, &vout, &vins, &meta);
    //             reports.insert(op.id, rep);
    //         }
    //         OpKind::Concat { axis } => {
    //             let mut vins: Vec<View> =
    //                 op.inputs.iter().map(|&t| view_from_tensor(st, t)).collect();
    //             let (vout, _changed) = apply_concat_with_unify(st, &mut vins, *axis);
    //             let rep = build_op_relabel_for_simple(st, op.id, &vout);
    //             reports.insert(op.id, rep);
    //         }
    //         OpKind::Split { axis, parts } => {
    //             let vin = view_from_tensor(st, op.inputs[0]);
    //             let vout = apply_split(st, &vin, *axis, parts);
    //             let rep = build_op_relabel_for_simple(st, op.id, &vout);
    //             reports.insert(op.id, rep);
    //         }
    //         OpKind::Merge { groups } => {
    //             let vin = view_from_tensor(st, op.inputs[0]);
    //             let vout = apply_merge(st, &vin, groups);
    //             let rep = build_op_relabel_for_simple(st, op.id, &vout);
    //             reports.insert(op.id, rep);
    //         }
    //         OpKind::Permute { perm } => {
    //             let vin = view_from_tensor(st, op.inputs[0]);
    //             let vout = apply_permute(st, &vin, perm);
    //             let rep = build_op_relabel_for_simple(st, op.id, &vout);
    //             reports.insert(op.id, rep);
    //         }
    //     }
    // }

    // ===== Build per-op relabel report from the final state (NON-mutating) =====
    let mut reports: HashMap<OpId, OpRelabel> = HashMap::new();
    for op in &g.operators {
        println!("op: {:?}", op);
        match &op.kind {
            OpKind::Einsum { spec } => {
                // Einsum은 contracted 라벨 메타가 필요하므로, 입력을 "coerce → normalize → unify"만 해서 메타를 재구성
                let mut vins: Vec<View> =
                    op.inputs.iter().map(|&t| view_from_tensor(st, t)).collect();
                let (vout, meta, _changed) = apply_einsum_with_unify(st, &mut vins, spec);
                let rep = build_op_relabel_for_einsum(st, op.id, &vout, &vins, &meta);
                reports.insert(op.id, rep);
            }
            // ⚠️ 나머지 연산자는 "최종 출력 텐서"를 그대로 사용해서 라벨만 부여 (연산 재적용 금지)
            _ => {
                let vout = view_from_tensor(st, op.outputs[0]);
                let rep = build_op_relabel_for_simple(st, op.id, &vout);
                reports.insert(op.id, rep);
            }
        }
    }

    reports
}

// 새 헬퍼: 입력 텐서들을 정규화하여 State에 write-back (변경 여부 리턴)
fn writeback_normalized_inputs(st: &mut State, op: &Operator) -> bool {
    let mut any = false;
    for &tin in &op.inputs {
        if let Some(orig) = st.tensors.get(&tin) {
            let mut t = orig.clone();
            // factor UF/children 변화를 반영해 축 분해를 leaf reps로 갱신
            st.normalize_tensor(&mut t);
            if update_tensor_if_changed(st, t) {
                any = true;
            }
        }
    }
    any
}

fn update_tensor_if_changed(st: &mut State, new_t: Tensor) -> bool {
    let changed = match st.tensors.get(&new_t.id) {
        None => true,
        Some(old) => old.shape != new_t.shape || old.decomp != new_t.decomp,
    };
    if changed {
        st.tensors.insert(new_t.id, new_t);
    }
    changed
}

/* Pretty-print per-op relabel reports without mutating State */
pub fn format_reports_pretty(st: &State, g: &Graph, reports: &HashMap<OpId, OpRelabel>) -> String {
    fn fmt_kind(k: &OpKind) -> String {
        match k {
            OpKind::Split { axis, parts } => format!("Split(axis={}, parts={:?})", axis, parts),
            OpKind::Merge { groups } => {
                let g = groups
                    .iter()
                    .map(|grp| format!("[{}]", grp.iter().map(|x| x.to_string()).join(",")))
                    .join(", ");
                format!("Merge(groups=[{}])", g)
            }
            OpKind::Permute { perm } => format!("Permute(perm={:?})", perm),
            OpKind::Concat { axis } => format!("Concat(axis={})", axis),
            OpKind::Einsum { spec } => format!("Einsum(\"{}\")", spec),
        }
    }

    fn fmt_factor(st: &State, fid: FactorId) -> String {
        let f = &st.factors[&fid];
        if f.is_concat {
            format!("cat{}={}", f.id, f.size)
        } else {
            format!("r{}={}", f.id, f.size)
        }
    }

    fn fmt_axis_labels(labels: &[usize]) -> String {
        format!("[{}]", labels.iter().map(|l| format!("L{}", l)).join(", "))
    }

    // leaf reps로 펼친 뒤, operator-local 라벨로 매핑 (전부 매핑되면 Some)
    fn map_axis_to_local_labels_ro(
        st: &State,
        axis_factors: &[FactorId],
        label_map: &HashMap<FactorId, usize>,
    ) -> Option<Vec<usize>> {
        let mut leafs: Vec<FactorId> = vec![];
        for &f in axis_factors {
            st.flatten_to_leaf_reps_ro(f, &mut leafs);
        }
        let mut out = Vec::with_capacity(leafs.len());
        for lf in leafs {
            let rep = st.uf_find_ro(lf);
            if let Some(&lbl) = label_map.get(&rep) {
                out.push(lbl);
            } else {
                return None;
            }
        }
        Some(out)
    }

    // Concat용: 입력 concat-축은 출력 cat-factor 하나의 라벨로 표시
    fn map_concat_input_axis_label_from_output(
        out_axis_labels: &[Vec<usize>],
        concat_axis: usize,
    ) -> Option<Vec<usize>> {
        out_axis_labels.get(concat_axis).cloned()
    }

    let mut s = String::new();

    // Prefer topo order if available; fall back to id order
    let order = g.topo_order();

    for opid in order {
        let op = &g.operators[opid];
        let rep = match reports.get(&op.id) {
            Some(r) => r,
            None => continue,
        };

        // Header
        s.push_str(&format!("=== op{}: {} ===\n", op.id, fmt_kind(&op.kind)));

        // I/O summary
        if !op.inputs.is_empty() {
            s.push_str("Inputs:\n");
            for (i, &tid) in op.inputs.iter().enumerate() {
                if let Some(t) = st.tensors.get(&tid) {
                    s.push_str(&format!("  - in#{i}  t{}  shape={:?}\n", t.id, t.shape));
                } else {
                    s.push_str(&format!("  - in#{i}  t{}  (missing)\n", tid));
                }
            }
        }
        if !op.outputs.is_empty() {
            s.push_str("Outputs:\n");
            for (i, &tid) in op.outputs.iter().enumerate() {
                if let Some(t) = st.tensors.get(&tid) {
                    s.push_str(&format!("  - out#{i} t{}  shape={:?}\n", t.id, t.shape));
                } else {
                    s.push_str(&format!("  - out#{i} t{}  (missing)\n", tid));
                }
            }
        }

        // Operator-local labels dictionary
        if !rep.factor_label_of.is_empty() {
            // invert: label -> factor
            let max_lbl = rep.factor_label_of.values().copied().max().unwrap_or(0);
            let mut by_lbl: Vec<Option<FactorId>> = vec![None; max_lbl + 1];
            for (fid, lbl) in &rep.factor_label_of {
                if *lbl >= by_lbl.len() {
                    by_lbl.resize(*lbl + 1, None);
                }
                by_lbl[*lbl] = Some(*fid);
            }
            s.push_str("Labels (operator-local):\n");
            for (lbl, maybe_fid) in by_lbl.iter().enumerate() {
                if let Some(fid) = maybe_fid {
                    s.push_str(&format!("  - L{} = {}\n", lbl, fmt_factor(st, *fid)));
                }
            }
        } else {
            s.push_str("Labels (operator-local): <none>\n");
        }

        // 입력 축 → 라벨 (항상 라벨만; 매핑 실패시 assert로 실패시켜 버그 조기 발견)
        if !op.inputs.is_empty() {
            s.push_str("Input axes:\n");
            for (i, &tid) in op.inputs.iter().enumerate() {
                let t = &st.tensors[&tid];
                for (ax, AxisDecomp(fs)) in t.decomp.iter().enumerate() {
                    // concat 입력 concat-축 특수 처리
                    if let OpKind::Concat { axis: cat_ax } = &op.kind {
                        if ax == *cat_ax {
                            if let Some(lbls) = map_concat_input_axis_label_from_output(
                                &rep.output_axis_labels,
                                *cat_ax,
                            ) {
                                s.push_str(&format!(
                                    "  - in#{} axis{} (len={}): {}\n",
                                    i,
                                    ax,
                                    t.shape[ax],
                                    fmt_axis_labels(&lbls)
                                ));
                                continue;
                            }
                        }
                    }
                    let lbls = map_axis_to_local_labels_ro(st, fs, &rep.factor_label_of)
                        .expect("pretty-print: input axis must be fully labeled");
                    s.push_str(&format!(
                        "  - in#{} axis{} (len={}): {}\n",
                        i,
                        ax,
                        t.shape[ax],
                        fmt_axis_labels(&lbls)
                    ));
                }
            }
        }

        // Output axis → labels
        if !rep.output_axis_labels.is_empty() {
            s.push_str("Output axes:\n");
            for (ax, lbls) in rep.output_axis_labels.iter().enumerate() {
                // axis length if available
                let axis_len = op
                    .outputs
                    .get(0)
                    .and_then(|tid| st.tensors.get(tid))
                    .and_then(|t| t.shape.get(ax).cloned());
                if let Some(len) = axis_len {
                    s.push_str(&format!(
                        "  - axis{} (len={}): {}\n",
                        ax,
                        len,
                        fmt_axis_labels(lbls)
                    ));
                } else {
                    s.push_str(&format!("  - axis{}: {}\n", ax, fmt_axis_labels(lbls)));
                }
            }
        }

        // Einsum: contracted axes per input
        if !rep.contracted_axis_labels.is_empty() {
            s.push_str("Contracted axes (einsum):\n");
            // group by input index
            let mut grouped: HashMap<usize, Vec<(usize, Vec<usize>)>> = HashMap::new();
            for (inp_idx, ax, lbls) in &rep.contracted_axis_labels {
                grouped
                    .entry(*inp_idx)
                    .or_default()
                    .push((*ax, lbls.clone()));
            }
            let mut keys: Vec<_> = grouped.keys().copied().collect();
            keys.sort_unstable();
            for k in keys {
                let mut axes = grouped.remove(&k).unwrap();
                axes.sort_by_key(|(ax, _)| *ax);
                for (ax, lbls) in axes {
                    // length if available
                    let len = op
                        .inputs
                        .get(k)
                        .and_then(|tid| st.tensors.get(tid))
                        .and_then(|t| t.shape.get(ax).cloned());
                    if let Some(l) = len {
                        s.push_str(&format!(
                            "  - in#{} axis{} (len={}): {}\n",
                            k,
                            ax,
                            l,
                            fmt_axis_labels(&lbls)
                        ));
                    } else {
                        s.push_str(&format!(
                            "  - in#{} axis{}: {}\n",
                            k,
                            ax,
                            fmt_axis_labels(&lbls)
                        ));
                    }
                }
            }
        }

        s.push('\n');
    }

    s
}

/* =========================
 * Example test with multi-root join
 * ========================= */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_root_join_fixpoint_relabel() {
        let mut st = State::new();
        let a = st.add_source_tensor(vec![8]);
        let b = st.add_source_tensor(vec![8]);

        let mut g = Graph::new();

        let t1 = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Split {
                axis: 0,
                parts: vec![2, 4],
            },
            vec![a],
            vec![t1],
        ); // op0

        let t2 = st.alloc_tensor_id();
        g.add_operator(OpKind::Permute { perm: vec![0] }, vec![b], vec![t2]); // op1

        let t3 = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Einsum {
                spec: "a,a->a".into(),
            },
            vec![t1, t2],
            vec![t3],
        ); // op2

        let t4 = st.alloc_tensor_id();
        g.add_operator(OpKind::Permute { perm: vec![0] }, vec![t3], vec![t4]); // op3

        let reports = relabel_graph(&mut st, &g);

        println!("reports: {:#?}", reports);

        // After fixpoint, t2's axis should be split to [2,4] (via unification at op2)
        let t2_view = &st.tensors[&t2];
        assert_eq!(
            t2_view.decomp[0].0.len(),
            2,
            "t2 should be refined to [2,4]"
        );

        // op2 (einsum) out axis should reuse the unified factors in order [2,4]
        let r2 = &reports[&2];
        assert_eq!(r2.output_axis_labels.len(), 1);
        assert_eq!(r2.output_axis_labels[0].len(), 2);

        // op0 (split) labels: output [2,4] → two labels
        let r0 = &reports[&0];
        assert_eq!(r0.output_axis_labels[0].len(), 1); // axis0=2
        assert_eq!(r0.output_axis_labels.len(), 2); // [2],[4] axes

        // op3 (permute) carries two labels as well
        let r3 = &reports[&3];
        assert_eq!(st.tensors[&t4].shape, vec![8]);
        assert_eq!(r3.output_axis_labels[0].len(), 2);

        println!("{}", format_reports_pretty(&st, &g, &reports));
    }

    #[test]
    fn mixed_pipeline_with_concat_and_einsum() {
        let mut st = State::new();
        let x = st.add_source_tensor(vec![4, 6]); // [4,6]
        let y = st.add_source_tensor(vec![6, 5]); // [6,5]
        let z = st.add_source_tensor(vec![4, 6]); // [4,6]

        let mut g = Graph::new();

        // op0: split x axis1: 6 -> [2,3] => [4,2,3]
        let t1 = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Split {
                axis: 1,
                parts: vec![2, 3],
            },
            vec![x],
            vec![t1],
        );

        // op1: concat(t1,z) along axis=0 => [[4,2,3] + [4,6]]  => unify non-concat axes
        let t2 = st.alloc_tensor_id();
        g.add_operator(OpKind::Concat { axis: 0 }, vec![t1, z], vec![t2]);

        // op2: einsum "abc,bc->ab"  with (t2,y) → unify 'b','c' across inputs
        let t3 = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Einsum {
                spec: "abc,bc->ab".into(),
            },
            vec![t2, y],
            vec![t3],
        );

        let reports = relabel_graph(&mut st, &g);

        // t2 (concat) should have cat factor on axis0; non-concat axes unified
        assert!(
            st.tensors[&t2].decomp[0].0.len() == 1
                && st.factors[&st.tensors[&t2].decomp[0].0[0]].is_concat
        );

        // einsum op2: output "ab" → two axes with labels; contracted 'c' is recorded
        let r2 = &reports[&2];
        assert_eq!(r2.output_axis_labels.len(), 2);
        assert!(!r2.contracted_axis_labels.is_empty());
    }

    /// Multi-root graph with reshape (split) + einsum join.
    ///
    /// Roots:
    ///   A: [12, 6]  --(Split axis0: [3,4])->  T1: [3,4,6]
    ///   B: [6, 5]   --(no-op Permute)------->  T2: [6,5]
    ///
    /// Join:
    ///   E: einsum("abc,cd->abd")(T1, T2)  =>  T3: [3,4,5]
    ///
    /// Checks:
    ///   - Shapes of T1, T2, T3
    ///   - 'c' axis unification: factor(T1 axis2) == factor(T2 axis0) after fixpoint write-back
    ///   - Einsum report has contracted axes info for both inputs
    #[test]
    fn multi_root_reshape_einsum() {
        let mut st = State::new();

        // Two independent roots
        let a = st.add_source_tensor(vec![12, 6]); // A: [x=12, c=6]
        let b = st.add_source_tensor(vec![6, 5]); // B: [c=6, d=5]

        let mut g = Graph::new();

        // op0: Split A axis0 -> [3,4,6]
        let t1 = st.alloc_tensor_id();
        let op0 = g.add_operator(
            OpKind::Split {
                axis: 0,
                parts: vec![3, 4],
            },
            vec![a],
            vec![t1],
        );

        // op1: No-op permute on B to create a separate root path op
        let t2 = st.alloc_tensor_id();
        let op1 = g.add_operator(OpKind::Permute { perm: vec![0, 1] }, vec![b], vec![t2]);

        // op2: Einsum join "abc,cd->abd"  (T1=[3,4,6], T2=[6,5]) -> T3=[3,4,5]
        let t3 = st.alloc_tensor_id();
        let op2 = g.add_operator(
            OpKind::Einsum {
                spec: "abc,cd->abd".into(),
            },
            vec![t1, t2],
            vec![t3],
        );

        // Run fixpoint relabel
        let reports = relabel_graph(&mut st, &g);

        // ===== Shape checks =====
        assert_eq!(
            st.tensors[&t1].shape,
            vec![3, 4, 6],
            "split result T1 shape mismatch"
        );
        assert_eq!(
            st.tensors[&t2].shape,
            vec![6, 5],
            "permute no-op T2 shape mismatch"
        );
        assert_eq!(
            st.tensors[&t3].shape,
            vec![3, 4, 5],
            "einsum output T3 shape mismatch"
        );

        // ===== Factor unification on 'c' =====
        // After fixpoint + write-back, T1 axis2 and T2 axis0 must share the same representative factor.
        let c_from_t1 = st.tensors[&t1].decomp[2].0[0];
        let c_from_t2 = st.tensors[&t2].decomp[0].0[0];
        // Compare union-find representatives to be robust
        let rep_t1 = st.uf_find(c_from_t1);
        let rep_t2 = st.uf_find(c_from_t2);
        assert_eq!(
            rep_t1, rep_t2,
            "einsum 'c' axes should be unified to the same factor rep"
        );

        // ===== Report checks (contracted 'c') =====
        // Einsum "abc,cd->abd" contracts 'c' from both inputs: expect entries for (0,2) and (1,0)
        let er = &reports[&op2];
        // Must have some contracted axes listed
        assert!(
            !er.contracted_axis_labels.is_empty(),
            "einsum report should include contracted axes"
        );
        // Gather (input_idx, axis_idx) pairs
        let mut got: Vec<(usize, usize)> = er
            .contracted_axis_labels
            .iter()
            .map(|(i, ax, _)| (*i, *ax))
            .collect();
        got.sort_unstable();
        assert_eq!(
            got,
            vec![(0, 2), (1, 0)],
            "expected contracted axes (0,2) and (1,0)"
        );

        // (Optional) also sanity-check operator-local output labels exist for the 3 output axes
        assert_eq!(
            er.output_axis_labels.len(),
            3,
            "einsum output should have 3 axes"
        );
        // axis0=3 should have some labels
        assert!(!er.output_axis_labels[0].is_empty());
        assert!(!er.output_axis_labels[1].is_empty());
        assert!(!er.output_axis_labels[2].is_empty());

        println!("{}", format_reports_pretty(&st, &g, &reports));
    }

    #[test]
    fn multi_root_two_einsums_with_reshape_and_transpose() {
        let mut st = State::new();

        // Roots
        let a = st.add_source_tensor(vec![12, 6]); // A: [12,6]
        let b = st.add_source_tensor(vec![6, 5]); // B: [6,5]
        let c = st.add_source_tensor(vec![5, 3, 2, 2]); // C: [5,3,2,2]

        let mut g = Graph::new();

        // op0: Split A axis0 -> [3,4,6]
        let t1 = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Split {
                axis: 0,
                parts: vec![3, 4],
            },
            vec![a],
            vec![t1],
        );

        // op1: Permute B (no-op) -> [6,5]
        let t2 = st.alloc_tensor_id();
        g.add_operator(OpKind::Permute { perm: vec![0, 1] }, vec![b], vec![t2]);

        // op2: Einsum1 "abc,cd->abd" : T1[3,4,6], T2[6,5] -> T3[3,4,5]
        let t3 = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Einsum {
                spec: "abc,cd->abd".into(),
            },
            vec![t1, t2],
            vec![t3],
        );

        // ===== Branch for einsum2 =====
        // op3: Transpose T3 to put 'd' first → Permute [2,0,1] : [a,b,d] -> [d,a,b] = [5,3,4]
        let t3p = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Permute {
                perm: vec![2, 0, 1],
            },
            vec![t3],
            vec![t3p],
        );

        // op4: Reshape C by merging last two axes: [[0],[1],[2,3]] -> [5,3,4]
        let c_merged = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Merge {
                groups: vec![vec![0], vec![1], vec![2, 3]],
            },
            vec![c],
            vec![c_merged],
        );

        // op5: Einsum2 with transpose already applied on left input:
        // spec "dab,dae->be"  where:
        //   left  = T3ᵖ [d=5, a=3, b=4]
        //   right = C'  [d=5, a=3, e=4]
        // contracts over d & a; output [b=4, e=4]
        let t_out = st.alloc_tensor_id();
        g.add_operator(
            OpKind::Einsum {
                spec: "dab,dae->be".into(),
            },
            vec![t3p, c_merged],
            vec![t_out],
        );

        // Run
        let reports = relabel_graph(&mut st, &g);

        // Shapes
        assert_eq!(st.tensors[&t1].shape, vec![3, 4, 6]); // split
        assert_eq!(st.tensors[&t2].shape, vec![6, 5]); // no-op perm
        assert_eq!(st.tensors[&t3].shape, vec![3, 4, 5]); // einsum1
        assert_eq!(st.tensors[&t3p].shape, vec![5, 3, 4]); // permute
        assert_eq!(st.tensors[&c_merged].shape, vec![5, 3, 4]); // merge
        assert_eq!(st.tensors[&t_out].shape, vec![4, 4]); // einsum2 result

        // Factor unifications for einsum2: 'd' and 'a' match across inputs
        let (d_l, a_l) = (
            st.tensors[&t3p].decomp[0].0[0],
            st.tensors[&t3p].decomp[1].0[0],
        );
        let (d_r, a_r) = (
            st.tensors[&c_merged].decomp[0].0[0],
            st.tensors[&c_merged].decomp[1].0[0],
        );
        assert_eq!(st.uf_find(d_l), st.uf_find(d_r), "einsum2: 'd' must unify");
        assert_eq!(st.uf_find(a_l), st.uf_find(a_r), "einsum2: 'a' must unify");

        // Report sanity for einsum2: contracted axes should include (0,0),(0,1),(1,0),(1,1)
        let r_e2 = &reports[&5];
        let mut contracted: Vec<(usize, usize)> = r_e2
            .contracted_axis_labels
            .iter()
            .map(|(i, ax, _)| (*i, *ax))
            .collect();
        contracted.sort_unstable();
        assert_eq!(contracted, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);

        println!("{}", format_reports_pretty(&st, &g, &reports));
    }
}
