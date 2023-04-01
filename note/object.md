# Object 类

`class Object` 在 TVM 中十分重要，基本上所有其他的类都直接或者间接继承自这个类。

```c++
class TVM_DLL Object {
 public:
  uint32_t type_index() const { return type_index_; }
  std::string GetTypeKey() const { return TypeIndex2Key(type_index_); }
  size_t GetTypeKeyHash() const { return TypeIndex2KeyHash(type_index_); }
  static std::string TypeIndex2Key(uint32_t tindex);
  static size_t TypeIndex2KeyHash(uint32_t tindex);
  static uint32_t TypeKey2Index(const std::string& key);
  static uint32_t _GetOrAllocRuntimeTypeIndex() { return TypeIndex::kRoot; }
  static uint32_t RuntimeTypeIndex() { return TypeIndex::kRoot; }
  inline int use_count() const;
 protected:
  uint32_t type_index_{0};
  RefCounterType ref_counter_{0};
  static uint32_t GetOrAllocRuntimeTypeIndex(const std::string& key, uint32_t static_tindex,
                                             uint32_t parent_tindex, uint32_t type_child_slots,
                                             bool type_child_slots_can_overflow);
}
```
如上所示，可以看到有一个很关键的概念就是 `type_index_` 这个概念，有一个全局唯一的表，里面记录了每一个 `Object` 对应的 `key` 和 `type`。

当一个 type 被第一次使用的时候，就会在运行时将 _type_index 注册到全局的表当中。`_type_index` 的值通过 `Object::GetOrAllocRuntimeTypeIndex` 这个接口来分配，具体的分配算法会用到下面的数据结构：
```c++
struct TypeInfo {};
class TypeContext {};
```
`TypeContext` 是个单例，具体的分配算法就是在父类预留的 `_type_child_slots` 范围内确定当前类的 `_type_index` ，然后更新 `type_table_` 这个vector，它的下标同时也是具体分配到的 `type_index`。

`type_index` 可以用于IsInstance这个辅助函数的加速，内部实现就是直接判断子类的_type_index的值是不是在父类预留的_type_child_slots范围之内

如果拿Object、ObjectPtr、ObjectRef这三个类和shared_ptr类比的话：

- Object相当于控制块，可以通过引用计数ref_counter_来控制对象的生命周期，对象的析构函数也可以通过deleter_这个函数指针指定
- Object的子类的除去Object基类的部分相当于数据块，里面保存有类的真实数据
- ObjectRef就像是shared_ptr这个wrapper，自身不包含实际数据，但是可以操作实际的数据
- ObjectPtr的作用在使用的角度有点类似ObjectRef，不同的是数据类型，ObjectPtr<T>是一个模板

## ObjectPtr

`ObjectPtr` 是指向 `Object` 的封装，行为类似指针，对各种操作符以及函数都进行了自己的实现

## ObjectRef
`ObjectRef` 是对 `ObjectPtr` 的进一步封装，具有类似智能指针的行为。

在 TVM 中，所有以Node为结尾的类名都是继承自Object，不以Node结尾的类名都是继承自ObjectRef，例如：

```c++
class Module : public ObjectRef {};
class ModuleNode : public Object {};
```
## make_object

```c++
template <typename T, typename... Args>
inline ObjectPtr<T> make_object(Args&&... args) {
  return SimpleObjAllocator().make_object<T>(std::forward<Args>(args)...);
}
```

可以看到，make_object内部调用了 `SimpleObjAllocator` 这个内存分配器。

make_object只是构造单个的Object，其实除了make_object之外，还有另外一个helper function， `make_inplace_array_object` 用于构造Object数组，同样是使用了 `SimpleObjAllocator` 这个内存分配器

### SimpleObjAllocator

SimpleObjAllocator继承自ObjAllocatorBase，两者的类关系定义如下：

```c++
template <typename Derived> 
class ObjAllocatorBase {};

class SimpleObjAllocator : public ObjAllocatorBase<SimpleObjAllocator> {};
```

这里用到了C++的一种编程技巧CRTP，它既可以实现静态多态，又可以复用代码，CRTP在TVM有多处应用

ObjAllocatorBase，它有两个成员函数，一个是构造单个Object的`make_object`，一个是构造Object数组的`make_inplace_array`

```c++
template <typename Derived>
class ObjAllocatorBase {
 public:
  /*!
   * \brief Make a new object using the allocator.
   * \tparam T The type to be allocated.
   * \tparam Args The constructor signature.
   * \param args The arguments.
   */
  template <typename T, typename... Args>
  inline ObjectPtr<T> make_object(Args&&... args) {
    using Handler = typename Derived::template Handler<T>;
    T* ptr = Handler::New(static_cast<Derived*>(this), std::forward<Args>(args)...);
    ptr->type_index_ = T::RuntimeTypeIndex();
    ptr->deleter_ = Handler::Deleter();
    return ObjectPtr<T>(ptr);
  }

  /*!
   * \tparam ArrayType The type to be allocated.
   * \tparam ElemType The type of array element.
   * \tparam Args The constructor signature.
   * \param num_elems The number of array elements.
   * \param args The arguments.
   */
  template <typename ArrayType, typename ElemType, typename... Args>
  inline ObjectPtr<ArrayType> make_inplace_array(size_t num_elems, Args&&... args) {
    using Handler = typename Derived::template ArrayHandler<ArrayType, ElemType>;
    ArrayType* ptr = Handler::New(static_cast<Derived*>(this), num_elems, std::forward<Args>(args)...);
    ptr->type_index_ = ArrayType::RuntimeTypeIndex();
    ptr->deleter_ = Handler::Deleter();
    return ObjectPtr<ArrayType>(ptr);
  }
};
```

`make_object`和`make_inplace_array`的处理流程相同，都是通过静态多态的方法调用了Derived这个子类定义的New函数来构造对象，同时把Derived这个子类中定义的Deleter这个删除器赋值给Object中定义的deleter_变量中，用于析构对象的时候用，下面是deleter_在Object中的定义：

```c++
class TVM_DLL Object {
 public:
  typedef void (*FDeleter)(Object* self);
 protected:
  FDeleter deleter_ = nullptr;
};
```

`SimpleObjAllocator`中主要实现了基类中调用的New和Deleter函数

```c++
class SimpleObjAllocator : public ObjAllocatorBase<SimpleObjAllocator> {
  // 调用ObjAllocatorBase::Handler::New 调用的实际是这里的实现，这就是静态多态
  template <typename T> 
  class Handler {
    using StorageType = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

    template <typename... Args>
    static T* New(SimpleObjAllocator*, Args&&... args) {
      StorageType* data = new StorageType();
      new (data) T(std::forward<Args>(args)...);
      return reinterpret_cast<T*>(data);
    }

    static Object::FDeleter Deleter() { return Deleter_; }
    static void Deleter_(Object* objptr) {
      T* tptr = static_cast<T*>(objptr);
      tptr->T::~T();
      delete reinterpret_cast<StorageType*>(tptr);
    }
  };
};
```

看到这里，所有的一切就真相大白了，`make_object`调用的New通过标准库的new来分配空间，然后再通过placement new来在分配的空间上构造对象。
删除器中先调用了析构函数，然后使用标准库的delete来释放内存

上面是`make_object`使用的New和Deleter，`make_inplace_array`所使用的New和Deleter的实现原理相同，只不过在分配和释放内存的时候，调用的是标准库的operator new/delete的数组版本


参考
- https://zhuanlan.zhihu.com/p/362281073
- https://zhuanlan.zhihu.com/p/406749650
- https://zhuanlan.zhihu.com/p/432851987