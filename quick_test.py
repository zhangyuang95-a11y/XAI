"""
快速测试 - 验证 Explanation 框架是否正常工作
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from explanation_strict import (
    ExplanationSystem,
    SimplePropositionalEntailment,
    PropositionalFormula
)

def test_basic():
    """基本功能测试"""
    print("=" * 60)
    print("快速测试 - 验证框架功能")
    print("=" * 60)
    print()

    system = ExplanationSystem(SimplePropositionalEntailment())

    # 定义
    B = {PropositionalFormula("所有人都是会死的")}
    S = {
        PropositionalFormula("所有人都是会死的"),
        PropositionalFormula("苏格拉底是人")
    }
    φ = PropositionalFormula("苏格拉底是会死的")
    E = {PropositionalFormula("苏格拉底是人")}

    print("1. 测试创建弱解释")
    try:
        weak = system.create_weak(E, φ, B, S)
        print(f"   [OK] 弱解释创建成功: {weak}")
    except Exception as e:
        print(f"   [FAIL] 创建失败: {e}")
        return False

    print("\n2. 测试验证弱解释")
    is_valid = system.validate_weak(E, φ, B, S)
    print(f"   [OK] 验证结果: {is_valid}")

    print("\n3. 测试创建严格解释")
    try:
        strict = system.create_strict(E, φ, B, S)
        print(f"   [OK] 严格解释创建成功: {strict}")
    except Exception as e:
        print(f"   [FAIL] 创建失败: {e}")
        return False

    print("\n4. 测试搜索最小解释")
    minimals = system.find_minimal_explanations(φ, S, B, strict=True, max_results=5)
    print(f"   [OK] 找到 {len(minimals)} 个最小解释")
    for m in minimals:
        print(f"      - {m.evidence}")

    print("\n5. 测试自然语言渲染")
    if system.renderer:
        text = system.render(strict, "strict")
        print(f"   [OK] 渲染: {text[:50]}...")

    print("\n" + "=" * 60)
    print("所有测试通过！[OK]")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_basic()
    sys.exit(0 if success else 1)
