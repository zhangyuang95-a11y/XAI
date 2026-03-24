\# 当前符号系统草稿



\## 1. 逻辑语言



设 L 为表示 agent 可用知识的形式语言。



\## 2. 证据空间



设



S = {e1, e2, ..., en}



表示 agent 在某一推理时刻可访问的全部证据集合。



其中每个 ei in L。



S 表示 agent 的 evidence space，而不是客观世界本身。



\## 3. why-question



设 q in L 表示需要被解释的目标命题。



q 是需要回答“为什么成立”的对象。



\## 4. explanation candidate



设 E subseteq S。



E 是一个候选 explanation。



\## 5. 语义蕴含



设 M 为 L 的模型类。



若对任意模型 m in M，只要 m 满足 E，就必然满足 q，

则称 E |= q。



\## 6. explanation



若 E subseteq S 且 E |= q，

则 E 是 q 的一个 explanation。



\## 7. minimal explanation



若 E |= q，并且对任意 e in E，

都有 E minus {e} |/= q，

则 E 是 minimal explanation。



\## 8. explanation function



定义 f(E, q) 为将形式 explanation 与问题转化为自然语言解释文本的函数。



注意：

\- E 是 explanation 的数学本体

\- f(E, q) 是其自然语言表达

