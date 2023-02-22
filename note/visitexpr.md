分析一下``Codegen``中调用的``VisitExpr``函数是如何执行的
```c++
LoweredOutput Codegen(IRModule mod, relay::Function func, String mod_name) {
    ...

    IRModule lowered_mod = tec::LowerTE(mod_name_, config_, [this](BaseFunc func) {
        // We need to maintain the constant map for external
        // functions so we pass this processing function which
        // allows us to process each function as we lower it.
        if (func->GetAttr<String>(attr::kCompiler).defined()) {
        UpdateConstants(func, &params_);
        }
        // TODO(@areusch, @jroesch): We should refactor this to
        // execute as a further pass, instead writing data to the
        // lowering process directly.
        tec::UpdateFunctionMetadata(func, this->function_metadata_);
    })(mod);

    Function lowered_main_func = Downcast<Function>(lowered_mod->Lookup("main"));

    heads_ = VisitExpr(lowered_main_func->body);
    ...

}
```

``VisitExpr``位于src/relay/backend/utils.h，``MemoizedExprTranslator::VisitExpr``

memo_中应该是一些之前获取到的Expr，这里可以看到会从memo_中查找Expr，如果有就直接返回。

```c++
  virtual OutputType VisitExpr(const Expr& n) {
    ICHECK(n.defined());
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return it->second;
    }
    auto res = BaseFunctor::VisitExpr(n);
    memo_[n] = res;
    return res;
  }

  using BaseFunctor = ::tvm::relay::ExprFunctor<OutputType(const Expr&)>;
```

``BaseFunctor``实际上是``ExprFunctor``类型，会调用``ExprFunctor::VisitExpr``，这个函数是一个虚函数。

```c++
template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
  ...
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;
  ...
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitExpr(const Expr& n, Args... args) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  ...
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_EXPR_FUNCTOR_DISPATCH(ConstantNode);
    ...
    return vtable;
  }

  #define RELAY_EXPR_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

};
```

InitVTable中调用NodeFunctor::set_dispatch接口，类型参数为tvm relay ir的各种表达式类型，
传入set_dispatch的函数参数是lambda函数，lambda函数体中执行self->VisitExpr_()。
self时传入的参数this，当从派生类中发起VisitExpr的时候，这个this将是派生类实例，而不是基类。

set_dispatch是tvm::NodeFunctor中的函数
NodeFunctor::set_dispatch是在函数指针表func_中添加传入的lamad函数，表项索引为类型参数的id

```c++
using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

template <typename R, typename... Args>
class NodeFunctor<R(const ObjectRef& n, Args...)> {

  template <typename TNode>
  TSelf& set_dispatch(FPointer f) {  // NOLINT(*)
    uint32_t tindex = TNode::RuntimeTypeIndex();
    if (func_.size() <= tindex) {
      func_.resize(tindex + 1, nullptr);
    }
    ICHECK(func_[tindex] == nullptr) << "Dispatch for " << TNode::_type_key << " is already set";
    func_[tindex] = f;
    return *this;
  }
}
```

InitVTable都调用完set_dispatch之后，将会返回NodeFunctor。
在VisitExpr中将会调用InitVTable返回的NodeFunctor，NodeFunctor的()操作符是经过重载的。

这里以传入的参数的类型id为索引，从func_表中获取对应的lambda函数体，并调用执行，也就是执行了类实例的VisitExpr_。
因为一般来说发起VisitExpr调用的是以tvm::relay::ExprFunctor为基类，并在VisitExpr_中完成业务操作的类，
所以这里VisitExpr_是调用的业务类中重载后的VisitExpr_方法。
业务类对自己关注的类型的VisitExpr_进行重载，在其中完成自己的操作。

如果派生类不对各种类型重载VisitExpr_，就会调用到tvm::relay::ExprFunctor定义的VisitExpr_，抛出异常

```c++
template <typename R, typename... Args>
class NodeFunctor<R(const ObjectRef& n, Args...)> {
  /*!
   * \brief invoke the functor, dispatch on type of n
   * \param n The Node argument
   * \param args The additional arguments
   * \return The result.
   */
    R operator()(const ObjectRef& n, Args... args) const {
        ICHECK(can_dispatch(n)) << "NodeFunctor calls un-registered function on type "
                            << n->GetTypeKey();
        return (*func_[n->type_index()])(n, std::forward<Args>(args)...);
    }
}
```