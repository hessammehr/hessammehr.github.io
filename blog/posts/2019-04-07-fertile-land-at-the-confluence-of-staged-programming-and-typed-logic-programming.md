# Fertile land at the confluence of staged programming and (typed) logic programming — Part 1

I recently started sketching out a logic programming library called
[Logic.jl](https://github.com/hessammehr/Logic.jl) (it really is very much a
sketch at the moment). As the name might imply, the library is implemented in
Julia, which seems like an odd choice of implementation language. Choosing
Julia and the twist on logic programming that I am aiming for are motivated by
a particularly interesting confluence of ideas that I have become aware of in
the past year and hope to describe in this blog post. I would love to hear
your thoughts of course; just keep in mind that I am not a computer scientist.  

## Idea #1: Staged programming and languages with first-class JIT compilation machinery

There is no shortage of programming languages with just-in-time (JIT)
accelerated virtual machines. Typical JITs are opaque: they step in at run
time and speed up your code without you having to tell them anything.
Increasingly, however, there are JITs of a different breed that, rather than
stay hidden and opaque, work _in conjunction_ with the program, giving rise to
a class of programming languages where application code can inspect and
influence the various stages of code lowering as a _precise_ mechanism for on-
demand code generation and behaviour adaptation. Using the notion of _staged
programming_ , the various components of the runtime monolith (type checker,
GC, codegen) can also be used in an _à la carte_ fashion, further blurring the
line between compile and run time. I find this paradigm superior to the AST-
level metaprogramming often encountered with LISP family languages. Where
conventional JITs like HotSpot are often used to bridge the performance gap
between bytecode-interpreted dynamically-typed languages like Python and
statically-typed compiled languages like OCaml, staged programming, e.g. in
Julia, also recovers some of the safety and expressive power of a
sophisticated type system in an otherwise dynamically-typed environment.  

The immediate trade-off is the overhead of including the full compiler
toolchain with application code. The run time memory footprint would then
resemble the equivalent of an AST/bytecode interpreter, libraries, and any
compiler passes, including [possibly] the type system and codegen (e.g. LLVM).
This is non-trivial and impractical in memory-constrained applications such
as embedded systems. In principle at least, one can ahead-of-time compile the
application and discard unused toolchain code as long as it can be inferred as
unnecessary. That said, annotating/inferring dependencies between application code and the various pieces of compiler/runtime might require some effort.

### Staged programming in the wild

Before we move on to Julia, let's look at a few examples of programming langauges/environments that feature elements of staged programming. I suspect that there are many more such systems out there that I am not aware of.

 [MetaOCaml](http://okmij.org/ftp/ML/MetaOCaml.html) is described as an OCaml dialect for multi-stage programming. OCaml seems like a particularly good choice of language in my opinion, because a lot of the power afforded by its type system is challenged when dealing with data the structure of which is unknown until run time. Here for example is a (simplified) definition of the `json` type from the OCaml package Yojson.

```ocaml
type json = [
  | `Assoc of (string * json) list
  | `Bool of bool
  | `Float of float
  | `Int of int
  | `List of json list
  | `Null
  | `String of string
]
```

Here is the Yojson package in use

```ocaml
let doc = 
```

 [Terra](http://terralang.org/), essentially a metaprogramming system using Lua and LLVM aimed at low-level system programming.

```lua
-- Terra allows the Lua interpreter and LLVM to interact.
-- Lua code can invoke LLVM for code generation.
-- LLVM can also call Lua, here to partially evaluate an expression.
local a = 5
terra sin5()
    return [ math.sin(a) ]
end

-- output bitcode
sin5:printpretty() 
> output:
> sin50 = terra() : {double}
>    return -0.95892427466314
> end

-- example code from terralang.org
```

### Staged programming in Julia

In my opinion Julia is the most successful implementation of the staged programming paradigm today. I say this based on the number of Julia users as well as how far it has taken the paradigm. 


In Julia functions act as the basic unit of JIT compilation, with types guiding the process through a language feature called _multiple dispatch_. Take the following simple function, for instance.

```julia
""" sum(col)
Return the sum of the elements of collection `col`
"""
function sum(col)
    result = zero(eltype(col))
    for elem in col
        result += elem
    end
end
```

No code generation happens for this function until it is invoked, e.g. ```sum([1,2,3])```, at which point the type of its argument ```Array{Int64, 1}``` recursively propagates through the body of the function. Much of the logic inside the function can be constant-folded given this concrete type.
far from perfect. 

The use of multiple dispatch as the primary driver of code generation in Julia seems to have worked out really well for the language. Still, I wonder if there are mechanisms that allow more expressive programming.

Julia was conceived as a programming language for high
performance numerical calculations but, unlike other languages in its league
like Matlab and Mathematica, its type system and compilation machinery appear
to be applicable well beyond the realm of scientific computing, as
demonstrated by projects like [imp](https://github.com/jamii/imp) and
[DataKnots.jl](https://github.com/rbt-lang/DataKnots.jl)  and
[Automa.jl](https://github.com/BioJulia/Automa.jl).

Staged-programming is also known as multi

## First-class embedding of logic programming  

Having learned the functional paradigm, many people find it hard to go back to
their old imperitive tools, which feel verbose and error-prone in comparison.
My brief exposure to logic programming has left me with a similar feeling
towards functional programming, namely the sense that logic programming allows
solving the problem in a more direct and natural way. Just as functional
programming seems removed from the physical reality of the computer because of
its pervasive use of the function as a layer of indirection, logic
programming's symbolic variables and predicates provide a further layer of
abstraction that allow the solution to be described by its properties rather
than its realization from a given set of inputs.

Many Prolog users are familiar with the less-than-ideal interoperability of
typical Prolog implementations with real-world code written in a language like
Python. In a follow-up blog post I will try to explain how Julia's multiple
dispatch and staged programming facilitate embedding logic programming to
bring most of its expressive power to Julia.
