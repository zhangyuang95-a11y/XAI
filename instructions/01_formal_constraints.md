# 形式化约束

本项目必须满足以下核心约束：

1. explanation 必须被定义为数学对象，而不是心理学概念或自然语言概念。
2. 自然语言 explanation 不是 explanation 的本体，只是形式 explanation 的表达结果。
3. explanation 的定义必须建立在语义蕴含关系上。
4. 不允许直接使用未定义符号。
5. 必须显式定义：
   - 逻辑语言 L
   - 模型类 M
   - 证据空间 S
   - 单个证据 e
   - why-question q
   - explanation candidate E
   - 语义蕴含 |=
   - explanation function f(E, q)

本项目中：

- explanation 的数学本体应为证据子集 E
- 若 E subseteq S 且 E |= q，则 E 是 q 的一个 explanation
- 自然语言 explanation 只是 f(E, q) 的输出

心理学解释、直觉解释、叙事解释、可读性解释只能作为 related work 或 comparison，不能作为 explanation 的定义基础。