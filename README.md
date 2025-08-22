# Tensor Graph Global Relabling

- multi_root_reshape_einsum
<img width="1775" height="780" alt="output (1)" src="https://github.com/user-attachments/assets/7a852b3e-f70a-4fa2-ba00-067a2f4a088d" />

```
Operator { id: 0, kind: Split { axis: 0, parts: [3, 4] }, inputs: [0], outputs: [2] }
Operator { id: 1, kind: Permute { perm: [0, 1] }, inputs: [1], outputs: [3] }
Operator { id: 2, kind: Einsum { spec: "abc,cd->abd" }, inputs: [2, 3], outputs: [4] }

=== op1: Permute(perm=[0, 1]) ===
Inputs:
  - in#0  t1  shape=[6, 5]
Outputs:
  - out#0 t3  shape=[6, 5]
Labels (operator-local):
  - L0 = r1=6
  - L1 = r3=5
Input axes:
  - in#0 axis0 (len=6): [L0]
  - in#0 axis1 (len=5): [L1]
Output axes:
  - axis0 (len=6): [L0]
  - axis1 (len=5): [L1]

=== op0: Split(axis=0, parts=[3, 4]) ===
Inputs:
  - in#0  t0  shape=[12, 6]
Outputs:
  - out#0 t2  shape=[3, 4, 6]
Labels (operator-local):
  - L0 = r4=3
  - L1 = r5=4
  - L2 = r1=6
Input axes:
  - in#0 axis0 (len=12): [L0, L1]
  - in#0 axis1 (len=6): [L2]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1]
  - axis2 (len=6): [L2]

=== op2: Einsum("abc,cd->abd") ===
Inputs:
  - in#0  t2  shape=[3, 4, 6]
  - in#1  t3  shape=[6, 5]
Outputs:
  - out#0 t4  shape=[3, 4, 5]
Labels (operator-local):
  - L0 = r4=3
  - L1 = r5=4
  - L2 = r3=5
  - L3 = r1=6
Input axes:
  - in#0 axis0 (len=3): [L0]
  - in#0 axis1 (len=4): [L1]
  - in#0 axis2 (len=6): [L3]
  - in#1 axis0 (len=6): [L3]
  - in#1 axis1 (len=5): [L2]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1]
  - axis2 (len=5): [L2]
Contracted axes (einsum):
  - in#0 axis2 (len=6): [L3]
  - in#1 axis0 (len=6): [L3]

```

- multi_root_two_einsums_with_reshape_and_transpose
<img width="2192" height="1019" alt="output" src="https://github.com/user-attachments/assets/ab1089d1-975c-46f4-9610-db40f2bb3ee4" />

```
Operator { id: 0, kind: Split { axis: 0, parts: [3, 4] }, inputs: [0], outputs: [3] }
Operator { id: 1, kind: Permute { perm: [0, 1] }, inputs: [1], outputs: [4] }
Operator { id: 2, kind: Einsum { spec: "abc,cd->abd" }, inputs: [3, 4], outputs: [5] }
Operator { id: 3, kind: Permute { perm: [2, 0, 1] }, inputs: [5], outputs: [6] }
Operator { id: 4, kind: Merge { groups: [[0], [1], [2, 3]] }, inputs: [2], outputs: [7] }
Operator { id: 5, kind: Einsum { spec: "dab,dae->be" }, inputs: [6, 7], outputs: [8] }

=== op4: Merge(groups=[[0], [1], [2,3]]) ===
Inputs:
  - in#0  t2  shape=[5, 3, 2, 2]
Outputs:
  - out#0 t7  shape=[5, 3, 4]
Labels (operator-local):
  - L0 = r3=5
  - L1 = r5=3
  - L2 = r6=2
  - L3 = r7=2
Input axes:
  - in#0 axis0 (len=5): [L0]
  - in#0 axis1 (len=3): [L1]
  - in#0 axis2 (len=2): [L2]
  - in#0 axis3 (len=2): [L3]
Output axes:
  - axis0 (len=5): [L0]
  - axis1 (len=3): [L1]
  - axis2 (len=4): [L2, L3]

=== op1: Permute(perm=[0, 1]) ===
Inputs:
  - in#0  t1  shape=[6, 5]
Outputs:
  - out#0 t4  shape=[6, 5]
Labels (operator-local):
  - L0 = r1=6
  - L1 = r3=5
Input axes:
  - in#0 axis0 (len=6): [L0]
  - in#0 axis1 (len=5): [L1]
Output axes:
  - axis0 (len=6): [L0]
  - axis1 (len=5): [L1]

=== op0: Split(axis=0, parts=[3, 4]) ===
Inputs:
  - in#0  t0  shape=[12, 6]
Outputs:
  - out#0 t3  shape=[3, 4, 6]
Labels (operator-local):
  - L0 = r5=3
  - L1 = r9=4
  - L2 = r1=6
Input axes:
  - in#0 axis0 (len=12): [L0, L1]
  - in#0 axis1 (len=6): [L2]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1]
  - axis2 (len=6): [L2]

=== op2: Einsum("abc,cd->abd") ===
Inputs:
  - in#0  t3  shape=[3, 4, 6]
  - in#1  t4  shape=[6, 5]
Outputs:
  - out#0 t5  shape=[3, 4, 5]
Labels (operator-local):
  - L0 = r5=3
  - L1 = r9=4
  - L2 = r3=5
  - L3 = r1=6
Input axes:
  - in#0 axis0 (len=3): [L0]
  - in#0 axis1 (len=4): [L1]
  - in#0 axis2 (len=6): [L3]
  - in#1 axis0 (len=6): [L3]
  - in#1 axis1 (len=5): [L2]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1]
  - axis2 (len=5): [L2]
Contracted axes (einsum):
  - in#0 axis2 (len=6): [L3]
  - in#1 axis0 (len=6): [L3]

=== op3: Permute(perm=[2, 0, 1]) ===
Inputs:
  - in#0  t5  shape=[3, 4, 5]
Outputs:
  - out#0 t6  shape=[5, 3, 4]
Labels (operator-local):
  - L0 = r3=5
  - L1 = r5=3
  - L2 = r9=4
Input axes:
  - in#0 axis0 (len=3): [L1]
  - in#0 axis1 (len=4): [L2]
  - in#0 axis2 (len=5): [L0]
Output axes:
  - axis0 (len=5): [L0]
  - axis1 (len=3): [L1]
  - axis2 (len=4): [L2]

=== op5: Einsum("dab,dae->be") ===
Inputs:
  - in#0  t6  shape=[5, 3, 4]
  - in#1  t7  shape=[5, 3, 4]
Outputs:
  - out#0 t8  shape=[4, 4]
Labels (operator-local):
  - L0 = r9=4
  - L1 = r6=2
  - L2 = r7=2
  - L3 = r3=5
  - L4 = r5=3
Input axes:
  - in#0 axis0 (len=5): [L3]
  - in#0 axis1 (len=3): [L4]
  - in#0 axis2 (len=4): [L0]
  - in#1 axis0 (len=5): [L3]
  - in#1 axis1 (len=3): [L4]
  - in#1 axis2 (len=4): [L1, L2]
Output axes:
  - axis0 (len=4): [L0]
  - axis1 (len=4): [L1, L2]
Contracted axes (einsum):
  - in#0 axis0 (len=5): [L3]
  - in#0 axis1 (len=3): [L4]
  - in#1 axis0 (len=5): [L3]
  - in#1 axis1 (len=3): [L4]
```
