## 如何在relay中添加一个算子

我们理下整个流程：

1. 当tvm中实现一个算子时，会调用 RELAY_REGISTER_OP进行注册；

2. 该注册会在 AttrRegistry<OpRegEntry, Op>（这是个单例模式的类）的entry_map_中加入一个OpRegEntry实例；

3. 而tvm处理一个外部输入的模型时，如果遇到这个算子，在Op::Get方法中从entry_map_表中读取对应的OpRegEntry实例：

```c++
#define RELAY_REGISTER_OP(OpName) TVM_REGISTER_OP(OpName)

// 这里定义了一个OpRegEntry的static引用变量，__COUNTER__宏保证这个变量名全局唯一
// static变量保证在main函数执行之前完成初始化
#define TVM_REGISTER_OP(OpName)                   \
  static OpRegEntry& __make_##Op__COUNTER__ =     \
    OpRegEntry::RegisterOrGet(OpName).set_name()



OpRegEntry& OpRegEntry::RegisterOrGet(const String& name) {
    return OpRegistry::Global()->RegisterOrGet(name);
}

using OpRegistry = AttrRegistry<OpRegEntry, Op>

static TSelf* Global() {
    static TSelf* inst = new TSelf();
    return inst;
}

using TSelf = AttrRegistry<EntryType, KeyType>;

// 在RegisterOrGet中，先是在entry_map_表中查找key为name表项是否存在；
// 如果存在，直接返回该表象的value；
// 如果不存在，就new一个EntryType类型实例，
// 然后用get获取类型指针（由entry_map_的定义倒推，eptr为EntryType*类型），
// 将这个数据实例加入到entry_map_表和entries_表中。

EntryType& RegisterOrGet(const String& name) {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end()) return *it->second;
    uint32_t registry_index = static_cast<uint32_t>(entries_.size());
    auto entry = std::unique_ptr<EntryType>(new EntryType(registry_index));
    auto* eptr = entry.get();
    eptr->name = name;
    entry_map_[name] = eptr;
    entries_.emplace_back(std::move(entry));
    return *eptr;
}

```

TVM_REGISTER_OP执行时，每注册一个op，都会创建OpRegEntry数据结构的一个实例

```c++
class OpRegEntry {
 public:
  const Op& op() const { return op_; }

  inline OpRegEntry& describe(const std::string& descr);  // NOLINT(*)
  inline OpRegEntry& add_argument(const std::string& name, const std::string& type,
                                  const std::string& description);
  inline OpRegEntry& add_type_rel(
      const std::string& rel_name,
      runtime::TypedPackedFunc<bool(const Array<Type>&, int, const Attrs&, const TypeReporter&)>
          type_rel_func);
  template <typename AttrsType>
  inline OpRegEntry& set_attrs_type();
  inline OpRegEntry& set_attrs_type_key(const String& key);
  inline OpRegEntry& set_num_inputs(int32_t n);  // NOLINT(*)
  inline OpRegEntry& set_support_level(int32_t level);  // NOLINT(*)

  template <typename ValueType>
  inline OpRegEntry& set_attr(const std::string& attr_name,  // NOLINT(*)
                              const ValueType& value, int plevel = 10);
  inline void reset_attr(const std::string& attr_name);

  // set the name of the op to be the same as registry
  inline OpRegEntry& set_name() {  // NOLINT(*)
    if (get()->name.length() == 0) {
      get()->name = name;
    }
    return *this;
  }
  /*!
   * \return the corresponding entry.
   */
  TVM_DLL static OpRegEntry& RegisterOrGet(const String& name);

 private:
  template <typename, typename>
  friend class AttrRegistry;
  std::string name;
  Op op_;
  // private constructor
  TVM_DLL OpRegEntry(uint32_t reg_index);
  // return internal pointer to op.
  inline OpNode* get();
  // update the attribute OpAttrMap
  TVM_DLL void UpdateAttr(const String& key, runtime::TVMRetValue value, int plevel);
};
```

AttrRegistry是一个模板类，不光op的注册是用的这个基础类，TVM中还有另外的两个模块的注册也是基于这个基础类：

```c++
using TargetTagRegistry = AttrRegistry<TargetTagRegEntry, TargetTag>
using TargetKindRegistry = AttrRegistry<TargetKindRegEntry, TargetKind>
```

AttrRegistry是一个单例，定义在src/node/attr_registry.h，主要数据结构和API如下（已将模板参数换成OpRegEntry和Op）：

```c++
class AttrRegistry {
public:
  // 通过创建一个OpRegEntry的方式注册一个op，
  OpRegEntry &RegisterOrGet(const String &name);
  // 获取维护在entries_中的已经创建过的OpRegEntry
  const OpRegEntry *Get(const String &name) const;
  // 获取所有创建过的op name
  Array<String> ListAllNames() const;
  
  void UpdateAttr(const String &attr_name, ...);
  // 这个类的一个单例
  static AttrRegistry *Global();

private:
  // entries_用于维护创建好的OpRegEntry对象的生命周期
  vector<unique_ptr<OpRegEntry>> entries_;
  // entry_map_使用map来快速使用创建过的OpRegEntry
  map<String, OpRegEntry *> entry_map_;
  // attrs_维护了op的属性
  map<String, unique_ptr<AttrRegistryMapContainerMap<Op>>> attrs_;
};


// RegisterOrGet接口被同名函数OpRegEntry::RegisterOrGet调用，它负责真正的创建一个OpRegEntry的实例并且维护起来
OpRegEntry& AttrRegistry::RegisterOrGet(const String& name) {
  if (entry_map_.find(name) != entry_map_.end()) 
    return entry_map_[name];
  auto entry = make_unique<OpRegEntry>(entries_.size());
  entry->name = name;
  entry_map_[name] = entry.get();
  entries_.emplace_back(move(entry));
  return entry_map_[name];
}
```

op的attribute会被维护在AttrRegistryMapContainerMap类型的对象中

```c++
class AttrRegistryMapContainerMap {
public:
  // 这个API是用于检查对于给定op，是否有对应的attribute被设置（函数名取的不好）
  int count(const Op &op) const;
  // 用来获取给定op的attribute
  const runtime::TVMRetValue& operator[](const Op& op) const;
  // 也是用来取给定op的attribute，只是带默认值
  // 这是个模板函数，为了简化代码直接改成了下面这样
  ValueType get(const Op &op, ValueType def_value) const;
​
private:
  // attr的名字
  String attr_name_;
  // 具体的attr value，这是个vector，其index对应注册op时创建OpRegEntry对象时使用的index，
  vector<pair<TVMRetValue, int>> data_;
};
```

[refer](https://zhuanlan.zhihu.com/p/369433448)
[refer](https://blog.csdn.net/zx_ros/article/details/123130815)
[refer](https://blog.csdn.net/zx_ros/article/details/123526147)