# IRModule 的数据结构

## 节点定义

树的节点定义在 `/include/tvm/relay/expr.h` 中，主要有以下几种类型：
- ConstantNode
- VarNode
- TupleNode
- CallNode
- LetNode
- IfNode

这些 Node 都继承了 `RelayExprNode`， `RelayExprNode` 又继承了 `BaseExprNode`

```c++
class IfNode : public ExprNode {
 public:
  Expr cond;
  Expr true_branch;
  Expr false_branch;
};

class CallNode : public ExprNode {
 public:

  Expr op;
  tvm::Array<relay::Expr> args;
  Attrs attrs;

};
```

以 IfNode 和 CallNode 为例看一下它们的实现

## 节点的数据访问

ExprVisitor 用于不修改程序而是执行程序分析和收集信息的 passes。ExprVisitor 继承自 ExprFunctor，ExprFunctor设置了VisitExpr_ 的虚函数。 ExprFunctor 提供了一个 public 接口方法 VisitExpr，它接受一个表达式和零个或多个参数并返回某种类型的实例。 需要为每种类型的 Expr override VisitExpr_ 的实现来定义 AST 遍历模式。

VisitExpr 和 VisitExpr_ 之间的关系与调度有关。 每个 VisitExpr_ 定义针对特定类型的表达式，但你并不总是知道你将访问节点是哪种类型。 为了解决这个问题，ExprFunctor提供了一个VisitExpr函数，它从给定的表达式路由到处理它的 VisitExpr_case。 尽管 C++ 已经提供了动态调度，但 ExprFunctor 定义了自己的 vtable，VisitExpr 使用它。 通过定义我们自己的 vtable，我们可以更好地控制调度。 例如，如果我们想定义一个PrintVisitor遍历器，在每次访问之前打印 “Here”，我们可以覆盖VisitExpr：

```c++
void PrintVisitor::VisitExpr(const Expr& expr) {
  std::cout << "Here" << std::endl;
  ExprFunctor::VisitExpr(expr);
}
```

ExprFunctor 本身是一个非常通用的类，这就是为什么会扩展ExprVisitor或ExprMutator的原因。 这些类扩展了ExprFunctor 并提供VisitExpr_的默认实现，用于捕获每个表达式类型的常见遍历模式。 拥有这些默认实现意味着我们只需要为需要不同行为的表达式类型提供进行重写VisitExpr_方法即可。

比如对于ConstantChecker这个类，就继承了ExprVisitor，并通过VisitExpr(expr)，访问数据。ExprVisitor的VisitExpr成员函数实现如下：

```c++
void ExprVisitor::VisitExpr(const Expr& expr) {
  auto it = visit_counter_.find(expr.get());
  if (it != visit_counter_.end()) {
    ++it->second;
  } else {
    using TParent = ExprFunctor<void(const Expr&)>;
    TParent::VisitExpr(expr);
    visit_counter_.insert({expr.get(), 1});
  }
}
```

可以看到这个类实际上调用的是父类 (ExprFunctor) 的VisitExpr，而ExprFunctor的VisitExpr的实现如下：

```c++
virtual R VisitExpr(const Expr& n, Args... args) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
```

可以看到ExprFunctor设置了VisitExpr虚函数，在解析时会回到ExprVisitor来解析节点，而ConstantChecker这个类继承了ExprVisitor，这样我们只需要在ConstantChecker类中重写VisitExpr_就可以了。

在ExprFunctor的VisitExpr实现中有一个RELAY_EXPR_FUNCTOR_DISPATCH宏，这个宏的定义如下：

```c++
#define RELAY_EXPR_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });
```

这里的self即为ExprFunctor的VisitExpr的实现中的vtable(n, this, std::forward<Args>(args)...)，而this指向ExprFunctor。又因为ExprVisitor::VisitExpr方法调用的是ExprFunctor的函数，所以这里的this指向的是ExprVisitor实例。

以 IfNode 为例子，看看ExprVisitor的VisitExpr_实现。由于this指向的是ExprVisitor实例，最后会在ExprVisitor实例中生成visit_counter_的列表。

```c++
void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}
```

visit_counter_是在ExprVisitor中定义的一个unordered_map，来标记在遍历 Relay AST 时某种 Expr 是否出现，同时记录下出现的次数。

```c++
std::unordered_map<const Object*, size_t> visit_counter_;
```

## 节点的修改

pass 是对 Relay 树结构，也可以说计算图进行优化，优化必然设计到对图结构的修改。这就是上面提到的ExprMutator子类，它和ExprVisitor一样继承自ExprFunctor。类的定义如下：

```c++
class ExprMutator : public ::tvm::relay::ExprFunctor<Expr(const Expr&)> {
 public:
  /*!
   * \brief Mutate is alias for VisitExpr
   * \return expr.
   */
  Expr Mutate(const Expr& expr) { return this->VisitExpr(expr); }
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const ConstantNode* op) override;
  Expr VisitExpr_(const GlobalVarNode* op) override;
  Expr VisitExpr_(const OpNode* op) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const CallNode* call_node) override;
  Expr VisitExpr_(const LetNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;
  Expr VisitExpr_(const TupleGetItemNode* op) override;
  Expr VisitExpr_(const RefCreate来表记Node* op) override;
  Expr VisitExpr_(const RefReadNode* op) override;
  Expr VisitExpr_(const RefWriteNode* op) override;
  Expr VisitExpr_(const ConstructorNode* op) override;
  Expr VisitExpr_(const MatchNode* op) override;

  /*!
   * \brief Used to visit the types inside of expressions.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);
  virtual Clause VisitClause(const Clause& c);
  virtual Pattern VisitPattern(const Pattern& c);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;
};
```
我们需要关注的是memo_这个成员变量，然后我们看一下这个类的VisitExpr实现：

```c++
Expr ExprMutator::VisitExpr(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = ExprFunctor::VisitExpr(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}
```

可以看到memo_存储了图中的各个节点。参考 IfNode 的实现：

```c++
Expr ExprMutator::VisitExpr_(const IfNode* op) {
  auto guard = this->Mutate(op->cond);
  auto true_b = this->Mutate(op->true_branch);
  auto false_b = this->Mutate(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}
```

如果 IFNode 的子节点都没有被修改，那么就返回这个节点本身。否则创建新的节点If(guard, true_b, false_b, op->span);并返回。这里构造新节点的类If的定义和实现分别在tvm/src/relay/ir/expr.h和tvm/src/relay/ir/expr.cc中：

```c++
class If : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param cond The condition of a if node.
   * \param true_branch The fall through branch
   * \param false_branch The branch for execution when condition is false.
   * \param span The source span of the expression.
   */
  TVM_DLL If(Expr cond, Expr true_branch, Expr false_branch, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(If, RelayExpr, IfNode);
};

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ObjectPtr<IfNode> n = make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  n->span = std::move(span);
  data_ = std::move(n);
}
```

参考
- http://www.giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E3%80%91%E4%B8%83%EF%BC%8C%E4%B8%87%E5%AD%97%E9%95%BF%E6%96%87%E5%85%A5%E9%97%A8TVM%20Pass/