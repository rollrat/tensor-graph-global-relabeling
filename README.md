# Tensor Graph Global Relabling

- reshape_to_composite_split_merge
<img width="1779" height="500" alt="output (2)" src="https://github.com/user-attachments/assets/6c92a2b8-4821-4953-af10-ae5d639f9693" />

```
op: Operator { id: 0, kind: ReshapeTo { out_shape: [4, 6, 3] }, inputs: [0], outputs: [1] }

=== op0: ReshapeTo([4, 6, 3]) ===
Inputs:
  - in#0  t0  shape=[12, 6]
Outputs:
  - out#0 t1  shape=[4, 6, 3]
Labels (operator-local):
  - L0 = r2=4
  - L1 = r3=3
  - L2 = r4=2
  - L3 = r5=3
Input axes:
  - in#0 axis0 (len=12): [L0, L1]
  - in#0 axis1 (len=6): [L2, L3]
Output axes:
  - axis0 (len=4): [L0]
  - axis1 (len=6): [L1, L2]
  - axis2 (len=3): [L3]
```

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

- reshape_and_two_einsums_example

<img width="1475" height="373" alt="re_autolayout" src="https://github.com/user-attachments/assets/971f27ed-e676-4217-8a4d-0e2ca85a94d4" />

```
Operator { id: 0, kind: ReshapeTo { out_shape: [4, 6, 3] }, inputs: [0], outputs: [3] }
Operator { id: 1, kind: ReshapeTo { out_shape: [6, 2, 8] }, inputs: [1], outputs: [4] }
Operator { id: 2, kind: Einsum { spec: "acb, cde -> abde" }, inputs: [3, 4], outputs: [5] }
Operator { id: 3, kind: Permute { perm: [1, 0, 2, 3] }, inputs: [5], outputs: [6] }
Operator { id: 4, kind: Einsum { spec: "fgde, eh -> fgdh" }, inputs: [6, 2], outputs: [7] }

=== op1: ReshapeTo([6, 2, 8]) ===
Inputs:
  - in#0  t1  shape=[6, 16]
Outputs:
  - out#0 t4  shape=[6, 2, 8]
Labels (operator-local):
  - L0 = r9=3
  - L1 = r10=2
  - L2 = r6=2
  - L3 = r4=8
Input axes:
  - in#0 axis0 (len=6): [L0, L1]
  - in#0 axis1 (len=16): [L2, L3]
Output axes:
  - axis0 (len=6): [L0, L1]
  - axis1 (len=2): [L2]
  - axis2 (len=8): [L3]

=== op0: ReshapeTo([4, 6, 3]) ===
Inputs:
  - in#0  t0  shape=[12, 6]
Outputs:
  - out#0 t3  shape=[4, 6, 3]
Labels (operator-local):
  - L0 = r8=4
  - L1 = r9=3
  - L2 = r10=2
  - L3 = r11=3
Input axes:
  - in#0 axis0 (len=12): [L0, L1]
  - in#0 axis1 (len=6): [L2, L3]
Output axes:
  - axis0 (len=4): [L0]
  - axis1 (len=6): [L1, L2]
  - axis2 (len=3): [L3]

=== op2: Einsum("acb, cde -> abde") ===
Inputs:
  - in#0  t3  shape=[4, 6, 3]
  - in#1  t4  shape=[6, 2, 8]
Outputs:
  - out#0 t5  shape=[4, 3, 2, 8]
Labels (operator-local):
  - L0 = r8=4
  - L1 = r11=3
  - L2 = r6=2
  - L3 = r4=8
  - L4 = r9=3
  - L5 = r10=2
Input axes:
  - in#0 axis0 (len=4): [L0]
  - in#0 axis1 (len=6): [L4, L5]
  - in#0 axis2 (len=3): [L1]
  - in#1 axis0 (len=6): [L4, L5]
  - in#1 axis1 (len=2): [L2]
  - in#1 axis2 (len=8): [L3]
Output axes:
  - axis0 (len=4): [L0]
  - axis1 (len=3): [L1]
  - axis2 (len=2): [L2]
  - axis3 (len=8): [L3]
Contracted axes (einsum):
  - in#0 axis1 (len=6): [L4, L5]
  - in#1 axis0 (len=6): [L4, L5]

=== op3: Permute(perm=[1, 0, 2, 3]) ===
Inputs:
  - in#0  t5  shape=[4, 3, 2, 8]
Outputs:
  - out#0 t6  shape=[3, 4, 2, 8]
Labels (operator-local):
  - L0 = r11=3
  - L1 = r8=4
  - L2 = r6=2
  - L3 = r4=8
Input axes:
  - in#0 axis0 (len=4): [L1]
  - in#0 axis1 (len=3): [L0]
  - in#0 axis2 (len=2): [L2]
  - in#0 axis3 (len=8): [L3]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1]
  - axis2 (len=2): [L2]
  - axis3 (len=8): [L3]

=== op4: Einsum("fgde, eh -> fgdh") ===
Inputs:
  - in#0  t6  shape=[3, 4, 2, 8]
  - in#1  t2  shape=[8, 5]
Outputs:
  - out#0 t7  shape=[3, 4, 2, 5]
Labels (operator-local):
  - L0 = r11=3
  - L1 = r8=4
  - L2 = r6=2
  - L3 = r5=5
  - L4 = r4=8
Input axes:
  - in#0 axis0 (len=3): [L0]
  - in#0 axis1 (len=4): [L1]
  - in#0 axis2 (len=2): [L2]
  - in#0 axis3 (len=8): [L4]
  - in#1 axis0 (len=8): [L4]
  - in#1 axis1 (len=5): [L3]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1]
  - axis2 (len=2): [L2]
  - axis3 (len=5): [L3]
Contracted axes (einsum):
  - in#0 axis3 (len=8): [L4]
  - in#1 axis0 (len=8): [L4]
```

- three_einsums_with_transposes_and_reshapes

<img width="2005" height="319" alt="graphviz" src="https://github.com/user-attachments/assets/01949741-e5e2-4bd2-903e-9b68b327be21" />


```
Operator { id: 0, kind: ReshapeTo { out_shape: [3, 4, 5, 2] }, inputs: [0], outputs: [4] }
Operator { id: 1, kind: Permute { perm: [0, 1, 3, 2] }, inputs: [4], outputs: [5] }
Operator { id: 2, kind: ReshapeTo { out_shape: [5, 2, 7] }, inputs: [1], outputs: [6] }
Operator { id: 3, kind: Einsum { spec: "abdc,cde->abe" }, inputs: [5, 6], outputs: [7] }
Operator { id: 4, kind: ReshapeTo { out_shape: [2, 6, 3, 4] }, inputs: [2], outputs: [8] }
Operator { id: 5, kind: Permute { perm: [1, 0, 2, 3] }, inputs: [8], outputs: [9] }
Operator { id: 6, kind: ReshapeTo { out_shape: [3, 5, 4] }, inputs: [3], outputs: [10] }
Operator { id: 7, kind: Permute { perm: [2, 0, 1] }, inputs: [10], outputs: [11] }
Operator { id: 8, kind: Einsum { spec: "gfhi,ihj->gfj" }, inputs: [9, 11], outputs: [12] }
Operator { id: 9, kind: ReshapeTo { out_shape: [3, 2, 2, 7] }, inputs: [7], outputs: [13] }
Operator { id: 10, kind: Permute { perm: [0, 2, 1, 3] }, inputs: [13], outputs: [14] }
Operator { id: 11, kind: ReshapeTo { out_shape: [3, 2, 2, 5] }, inputs: [12], outputs: [15] }
Operator { id: 12, kind: Permute { perm: [1, 0, 2, 3] }, inputs: [15], outputs: [16] }
Operator { id: 13, kind: Einsum { spec: "axfy,fazt->xyt" }, inputs: [14, 16], outputs: [17] }

=== op6: ReshapeTo([3, 5, 4]) ===
Inputs:
  - in#0  t3  shape=[15, 4]
Outputs:
  - out#0 t10  shape=[3, 5, 4]
Labels (operator-local):
  - L0 = r8=3
  - L1 = r9=5
  - L2 = r7=4
Input axes:
  - in#0 axis0 (len=15): [L0, L1]
  - in#0 axis1 (len=4): [L2]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=5): [L1]
  - axis2 (len=4): [L2]

=== op7: Permute(perm=[2, 0, 1]) ===
Inputs:
  - in#0  t10  shape=[3, 5, 4]
Outputs:
  - out#0 t11  shape=[4, 3, 5]
Labels (operator-local):
  - L0 = r7=4
  - L1 = r8=3
  - L2 = r9=5
Input axes:
  - in#0 axis0 (len=3): [L1]
  - in#0 axis1 (len=5): [L2]
  - in#0 axis2 (len=4): [L0]
Output axes:
  - axis0 (len=4): [L0]
  - axis1 (len=3): [L1]
  - axis2 (len=5): [L2]

=== op4: ReshapeTo([2, 6, 3, 4]) ===
Inputs:
  - in#0  t2  shape=[12, 12]
Outputs:
  - out#0 t8  shape=[2, 6, 3, 4]
Labels (operator-local):
  - L0 = r10=2
  - L1 = r14=3
  - L2 = r15=2
  - L3 = r8=3
  - L4 = r7=4
Input axes:
  - in#0 axis0 (len=12): [L0, L1, L2]
  - in#0 axis1 (len=12): [L3, L4]
Output axes:
  - axis0 (len=2): [L0]
  - axis1 (len=6): [L1, L2]
  - axis2 (len=3): [L3]
  - axis3 (len=4): [L4]

=== op5: Permute(perm=[1, 0, 2, 3]) ===
Inputs:
  - in#0  t8  shape=[2, 6, 3, 4]
Outputs:
  - out#0 t9  shape=[6, 2, 3, 4]
Labels (operator-local):
  - L0 = r14=3
  - L1 = r15=2
  - L2 = r10=2
  - L3 = r8=3
  - L4 = r7=4
Input axes:
  - in#0 axis0 (len=2): [L2]
  - in#0 axis1 (len=6): [L0, L1]
  - in#0 axis2 (len=3): [L3]
  - in#0 axis3 (len=4): [L4]
Output axes:
  - axis0 (len=6): [L0, L1]
  - axis1 (len=2): [L2]
  - axis2 (len=3): [L3]
  - axis3 (len=4): [L4]

=== op8: Einsum("gfhi,ihj->gfj") ===
Inputs:
  - in#0  t9  shape=[6, 2, 3, 4]
  - in#1  t11  shape=[4, 3, 5]
Outputs:
  - out#0 t12  shape=[6, 2, 5]
Labels (operator-local):
  - L0 = r14=3
  - L1 = r15=2
  - L2 = r10=2
  - L3 = r9=5
  - L4 = r8=3
  - L5 = r7=4
Input axes:
  - in#0 axis0 (len=6): [L0, L1]
  - in#0 axis1 (len=2): [L2]
  - in#0 axis2 (len=3): [L4]
  - in#0 axis3 (len=4): [L5]
  - in#1 axis0 (len=4): [L5]
  - in#1 axis1 (len=3): [L4]
  - in#1 axis2 (len=5): [L3]
Output axes:
  - axis0 (len=6): [L0, L1]
  - axis1 (len=2): [L2]
  - axis2 (len=5): [L3]
Contracted axes (einsum):
  - in#0 axis2 (len=3): [L4]
  - in#0 axis3 (len=4): [L5]
  - in#1 axis0 (len=4): [L5]
  - in#1 axis1 (len=3): [L4]

=== op11: ReshapeTo([3, 2, 2, 5]) ===
Inputs:
  - in#0  t12  shape=[6, 2, 5]
Outputs:
  - out#0 t15  shape=[3, 2, 2, 5]
Labels (operator-local):
  - L0 = r14=3
  - L1 = r15=2
  - L2 = r10=2
  - L3 = r9=5
Input axes:
  - in#0 axis0 (len=6): [L0, L1]
  - in#0 axis1 (len=2): [L2]
  - in#0 axis2 (len=5): [L3]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=2): [L1]
  - axis2 (len=2): [L2]
  - axis3 (len=5): [L3]

=== op12: Permute(perm=[1, 0, 2, 3]) ===
Inputs:
  - in#0  t15  shape=[3, 2, 2, 5]
Outputs:
  - out#0 t16  shape=[2, 3, 2, 5]
Labels (operator-local):
  - L0 = r15=2
  - L1 = r14=3
  - L2 = r10=2
  - L3 = r9=5
Input axes:
  - in#0 axis0 (len=3): [L1]
  - in#0 axis1 (len=2): [L0]
  - in#0 axis2 (len=2): [L2]
  - in#0 axis3 (len=5): [L3]
Output axes:
  - axis0 (len=2): [L0]
  - axis1 (len=3): [L1]
  - axis2 (len=2): [L2]
  - axis3 (len=5): [L3]

=== op2: ReshapeTo([5, 2, 7]) ===
Inputs:
  - in#0  t1  shape=[5, 14]
Outputs:
  - out#0 t6  shape=[5, 2, 7]
Labels (operator-local):
  - L0 = r2=5
  - L1 = r16=2
  - L2 = r17=7
Input axes:
  - in#0 axis0 (len=5): [L0]
  - in#0 axis1 (len=14): [L1, L2]
Output axes:
  - axis0 (len=5): [L0]
  - axis1 (len=2): [L1]
  - axis2 (len=7): [L2]

=== op0: ReshapeTo([3, 4, 5, 2]) ===
Inputs:
  - in#0  t0  shape=[12, 10]
Outputs:
  - out#0 t4  shape=[3, 4, 5, 2]
Labels (operator-local):
  - L0 = r14=3
  - L1 = r15=2
  - L2 = r23=2
  - L3 = r2=5
  - L4 = r16=2
Input axes:
  - in#0 axis0 (len=12): [L0, L1, L2]
  - in#0 axis1 (len=10): [L3, L4]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1, L2]
  - axis2 (len=5): [L3]
  - axis3 (len=2): [L4]

=== op1: Permute(perm=[0, 1, 3, 2]) ===
Inputs:
  - in#0  t4  shape=[3, 4, 5, 2]
Outputs:
  - out#0 t5  shape=[3, 4, 2, 5]
Labels (operator-local):
  - L0 = r14=3
  - L1 = r15=2
  - L2 = r23=2
  - L3 = r16=2
  - L4 = r2=5
Input axes:
  - in#0 axis0 (len=3): [L0]
  - in#0 axis1 (len=4): [L1, L2]
  - in#0 axis2 (len=5): [L4]
  - in#0 axis3 (len=2): [L3]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1, L2]
  - axis2 (len=2): [L3]
  - axis3 (len=5): [L4]

=== op3: Einsum("abdc,cde->abe") ===
Inputs:
  - in#0  t5  shape=[3, 4, 2, 5]
  - in#1  t6  shape=[5, 2, 7]
Outputs:
  - out#0 t7  shape=[3, 4, 7]
Labels (operator-local):
  - L0 = r14=3
  - L1 = r15=2
  - L2 = r23=2
  - L3 = r17=7
  - L4 = r16=2
  - L5 = r2=5
Input axes:
  - in#0 axis0 (len=3): [L0]
  - in#0 axis1 (len=4): [L1, L2]
  - in#0 axis2 (len=2): [L4]
  - in#0 axis3 (len=5): [L5]
  - in#1 axis0 (len=5): [L5]
  - in#1 axis1 (len=2): [L4]
  - in#1 axis2 (len=7): [L3]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=4): [L1, L2]
  - axis2 (len=7): [L3]
Contracted axes (einsum):
  - in#0 axis2 (len=2): [L4]
  - in#0 axis3 (len=5): [L5]
  - in#1 axis0 (len=5): [L5]
  - in#1 axis1 (len=2): [L4]

=== op9: ReshapeTo([3, 2, 2, 7]) ===
Inputs:
  - in#0  t7  shape=[3, 4, 7]
Outputs:
  - out#0 t13  shape=[3, 2, 2, 7]
Labels (operator-local):
  - L0 = r14=3
  - L1 = r15=2
  - L2 = r23=2
  - L3 = r17=7
Input axes:
  - in#0 axis0 (len=3): [L0]
  - in#0 axis1 (len=4): [L1, L2]
  - in#0 axis2 (len=7): [L3]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=2): [L1]
  - axis2 (len=2): [L2]
  - axis3 (len=7): [L3]

=== op10: Permute(perm=[0, 2, 1, 3]) ===
Inputs:
  - in#0  t13  shape=[3, 2, 2, 7]
Outputs:
  - out#0 t14  shape=[3, 2, 2, 7]
Labels (operator-local):
  - L0 = r14=3
  - L1 = r23=2
  - L2 = r15=2
  - L3 = r17=7
Input axes:
  - in#0 axis0 (len=3): [L0]
  - in#0 axis1 (len=2): [L2]
  - in#0 axis2 (len=2): [L1]
  - in#0 axis3 (len=7): [L3]
Output axes:
  - axis0 (len=3): [L0]
  - axis1 (len=2): [L1]
  - axis2 (len=2): [L2]
  - axis3 (len=7): [L3]

=== op13: Einsum("axfy,fazt->xyt") ===
Inputs:
  - in#0  t14  shape=[3, 2, 2, 7]
  - in#1  t16  shape=[2, 3, 2, 5]
Outputs:
  - out#0 t17  shape=[2, 7, 5]
Labels (operator-local):
  - L0 = r23=2
  - L1 = r17=7
  - L2 = r9=5
  - L3 = r14=3
  - L4 = r15=2
  - L5 = r10=2
Input axes:
  - in#0 axis0 (len=3): [L3]
  - in#0 axis1 (len=2): [L0]
  - in#0 axis2 (len=2): [L4]
  - in#0 axis3 (len=7): [L1]
  - in#1 axis0 (len=2): [L4]
  - in#1 axis1 (len=3): [L3]
  - in#1 axis2 (len=2): [L5]
  - in#1 axis3 (len=5): [L2]
Output axes:
  - axis0 (len=2): [L0]
  - axis1 (len=7): [L1]
  - axis2 (len=5): [L2]
Contracted axes (einsum):
  - in#0 axis0 (len=3): [L3]
  - in#0 axis2 (len=2): [L4]
  - in#1 axis0 (len=2): [L4]
  - in#1 axis1 (len=3): [L3]
  - in#1 axis2 (len=2): [L5]
```
