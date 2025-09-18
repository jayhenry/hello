#!/usr/bin/env python3
"""
简化版 Pydantic Schema 生成流程演示
"""

from pydantic import BaseModel, ConfigDict, ValidationError, InstanceOf
from typing import Any

print("=== Pydantic Schema 生成和验证流程 ===\n")

# 1. 基本 Schema 生成
print("1. 基本 Schema 生成过程:")
class Person(BaseModel):
    name: str
    age: int
    email: str | None = None

print("Person 模型的核心信息:")
print(f"- 字段: {list(Person.model_fields.keys())}")
print(f"- Schema 类型: {Person.__pydantic_core_schema__['type']}")
print()

# 2. 验证过程演示
print("2. 数据验证过程:")
try:
    # 正确数据
    person1 = Person(name="张三", age=25, email="zhangsan@example.com")
    print(f"✓ 验证成功: {person1}")
    
    # 类型转换
    person2 = Person(name="李四", age="30")  # age 是字符串但可以转换
    print(f"✓ 类型转换: {person2}")
    
    # 验证失败
    person3 = Person(name="王五", age="invalid_age")
except ValidationError as e:
    print(f"✗ 验证失败: {e}")
print()

# 3. Schema 的实际结构
print("3. Schema 的实际结构 (简化版):")
schema = Person.__pydantic_core_schema__
fields_schema = schema['schema']['fields']
for field_name, field_info in fields_schema.items():
    field_type = field_info['schema'].get('type', 'unknown')
    required = field_info.get('type') == 'model-field'
    print(f"- {field_name}: {field_type} ({'required' if required else 'optional'})")
print()

# 4. arbitrary_types_allowed 演示
print("4. arbitrary_types_allowed 的影响:")

class CustomClass:
    def __init__(self, value: str):
        self.value = value
    
    def __str__(self):
        return f"CustomClass({self.value})"

# 不允许任意类型的模型
try:
    class ModelWithCustomClass1(BaseModel):
        name: str
        custom_obj: CustomClass  # 未允许任意类型，这里会报错。因为 CustomClass 没有 __get_pydantic_core_schema__ 方法，无法提供schema
except Exception as e:
    print(f"✗ ModelWithCustomClass1 初始化失败: {e}")

class ModelWithCustomClass2(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    custom_obj: CustomClass  # 允许任意类型，不会报错

print()
inst2 = ModelWithCustomClass2(name="测试2", custom_obj=CustomClass("inst2"))
print(f"✓ ModelWithCustomClass2 初始化成功: {inst2}")

try:
    inst2_2 = ModelWithCustomClass2(name="测试2_2", custom_obj="inst2_2")
except Exception as e:
    print("虽然允许任意类型，以让ModelWithCustomClass2正常定义，但是运行时仍然会做类型验证，传入非 CustomClass 实例会报错")
    print(f"✗ ModelWithCustomClass2 允许任意类型初始化失败: {e}")
print()

class ModelWithCustomClass3(BaseModel):
    name: str
    custom_obj: InstanceOf[CustomClass]  # 使用 InstanceOf 来限制类型，不会报错

inst3 = ModelWithCustomClass3(name="测试3", custom_obj=CustomClass("inst3"))
print(f"✓ ModelWithCustomClass3 初始化成功: {inst3}")
print()

# 5. 允许任意类型的第2个演示
print("5. 允许任意类型的第2个演示:")
# 允许任意类型的模型
class ModelWithArbitraryTypes(BaseModel):
    # 没有这行，在这里定义时就会报错，因为 custom_obj 是 Any 类型，不能提供schema（即 没有 __get_pydantic_core_schema__ 方法）
    model_config = ConfigDict(arbitrary_types_allowed=True)  
    name: str
    custom_obj: Any

print("✓ 成功创建允许任意类型的模型")

# 创建实例
custom = CustomClass("测试值")
instance = ModelWithArbitraryTypes(name="测试", custom_obj=custom)
print(f"✓ 实例创建成功: name={instance.name}, custom_obj={instance.custom_obj}")

# 允许任意类型：无论传什么都能成功
print("✓ 允许任意类型的模型：以下不同类型均能成功初始化")
values_to_test = [
    "这不是 CustomClass 实例",
    42,
    {"k": "v"},
    [1, 2, 3],
    None,
    custom,
]
for val in values_to_test:
    obj_any = ModelWithArbitraryTypes(name="测试Any", custom_obj=val)
    print(f"  - 传入 {type(val).__name__}: custom_obj={obj_any.custom_obj}")
print()

# 5. Schema 的作用总结
print("5. Schema 的作用总结:")
print("✓ 类型验证：确保数据符合定义的类型")
print("✓ 数据转换：自动进行合理的类型转换")
print("✓ 默认值：处理可选字段的默认值")
print("✓ 嵌套验证：递归验证复杂数据结构")
print("✓ 错误报告：提供详细的验证错误信息")
print("✓ 序列化：支持模型与字典/JSON 的相互转换")
