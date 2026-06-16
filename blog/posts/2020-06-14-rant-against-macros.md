# A rant against macros

I used to be a huge fan of macros. I remember reading SICP and being amazed that you could use the language to generate and transform code. How cool is that? First a couple of examples: Clojure's `core.async` library includes a `go` macro that lets you launch goroutine-like tasks without having to change the language.

```clojure
; https://github.com/clojure/core.async/blob/master/examples/walkthrough.clj
(let [c1 (chan)
      c2 (chan)]
  (go (while true
        (let [[v ch] (alts! [c1 c2])]
          (println "Read" v "from" ch))))
  (go (>! c1 "hi"))
  (go (>! c2 "there")))
```

The Turing library lets you write probabilistic programs in Julia as if you're using a dedicated probabilistic programming language (PPL):

```julia
@model gdemo(x, y) = begin
    # Assumptions
    σ ~ InverseGamma(2,3)
    μ ~ Normal(0,sqrt(σ))
    # Observations
    x ~ Normal(μ, sqrt(σ))
    y ~ Normal(μ, sqrt(σ))
end
```

Fast forward to 2018 when I sat down with Chris Rackauckas before JuliaCon and he mentioned he'd been in touch with the Turing developers. I thought he bring up their PPL syntax and how it's so wonderful that Julia lets you mold the language, but when I prompted him he said the macros have gotten in the way of using Turing as a library. He said functions and types were the way forward if you want things to compose.

Since then, I've written a couple of macros of my own and, powerful as they are, I have come to the conclusion that the problems I used them for were better handled by i) new or more expressive data structures, ii) plain old functions, iii) accepting a small amount of extra verbosity. In return you get, i) better interoperability, ii) code that is more explicit and easier to undestand, iii) much easier debugging, iv) a more robust design, v) much better support from your tools (_e.g._ IDE, REPL).

Let's look at a simpler model:

```julia
using Distributions

@model normal_model(x) = begin
    # just a simple transformation; z is still observed, just like x
    z = 2x
    # sample y
    y ~ Normal(0.0, 1.0)
    # observe z
    z ~ Normal(y, 1.0)
end
```

This is what it expands to

```julia
quote
    #= /home/group/.julia/packages/DynamicPPL/9OFG0/src/compiler.jl:348 =#
    function var"##evaluator#371"(_rng::Random.AbstractRNG, _model::DynamicPPL.Model, _varinfo::DynamicPPL.AbstractVarInfo, _sampler::AbstractMCMC.AbstractSampler, _context::DynamicPPL.AbstractContext)
        #= /home/group/.julia/packages/DynamicPPL/9OFG0/src/compiler.jl:355 =#
        begin
            x = (DynamicPPL.matchingvalue)(_sampler, _varinfo, _model.args.x)
        end
        #= /home/group/.julia/packages/DynamicPPL/9OFG0/src/compiler.jl:356 =#
        begin
            #= REPL[22]:1 =#
            #= REPL[22]:2 =#
            z = 2x
            #= REPL[22]:3 =#
            begin
                var"##tmpright#363" = Normal(0.0, 1.0)
                var"##tmpright#363" isa Union{Distribution, AbstractVector{<:Distribution}} || throw(ArgumentError("Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions."))
                var"##vn#365" = y
                var"##inds#366" = ()
                y = (DynamicPPL.tilde_assume)(_rng, _context, _sampler, var"##tmpright#363", var"##vn#365", var"##inds#366", _varinfo)
            end
            #= REPL[22]:4 =#
            begin
                var"##tmpright#367" = Normal(y, 1.0)
                var"##tmpright#367" isa Union{Distribution, AbstractVector{<:Distribution}} || throw(ArgumentError("Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions."))
                var"##vn#369" = z
                var"##inds#370" = ()
                z = (DynamicPPL.tilde_assume)(_rng, _context, _sampler, var"##tmpright#367", var"##vn#369", var"##inds#370", _varinfo)
            end
        end
    end
    #= /home/group/.julia/packages/DynamicPPL/9OFG0/src/compiler.jl:359 =#
    var"##generator#372"(x) = begin
            #= /home/group/.julia/packages/DynamicPPL/9OFG0/src/compiler.jl:359 =#
            (DynamicPPL.Model)(var"##evaluator#371", (DynamicPPL.namedtuple)(NamedTuple{(:x,), Tuple{Core.Typeof(x)}}, (x,)), (DynamicPPL.ModelGen){(:x,)}(var"##generator#372", NamedTuple()))
        end
    #= /home/group/.julia/packages/DynamicPPL/9OFG0/src/compiler.jl:360 =#
    var"##generator#372"(; x) = begin
            #= /home/group/.julia/packages/DynamicPPL/9OFG0/src/compiler.jl:344 =#
            var"##generator#372"(x)
        end
    #= /home/group/.julia/packages/DynamicPPL/9OFG0/src/compiler.jl:362 =#
    begin
        $(Expr(:meta, :doc))
        normal_model = (DynamicPPL.ModelGen){(:x,)}(var"##generator#372", NamedTuple())
    end
end
```

And now to sample it:
```julia
sample(normal_model(3.0), NUTS(), 1000)

# Summary Statistics
#   parameters    mean     std  naive_se    mcse       ess   r_hat
#   ──────────  ──────  ──────  ────────  ──────  ────────  ──────
#            y  0.0096  1.0146    0.0454  0.0978  168.5831  0.9986
#            z  0.0169  1.4692    0.0657  0.1204  158.7592  0.9992
```

And we see that Turing has sampled both `y` and `z`, where `z` should have been marked as deterministic and observed rather than sampled. Now, I'm sure this is well-documented somewhere but the point is that when you use a macro, your Julia code no longer functions the way you would expect. Worse, yet finding out why means being able to navigate the mess of generated symbols in the expanded version. And yes, the authors can fix this (if it's actually a bug) but it doesn't change the problem that the language inside that block is no longer Julia. You keep having to second guess yourself every time you reach for a new language feature.

Increasingly, macros, even nice hygienic ones remind me of the horrible mess that's C/C++ macros: an untamed partial language with its own semantics that you need to learn and use, and how people have created whole programming languages in part to escape this ugly metalangauge problem. It's true that homoiconic languages mostly get rid of the macro/preprocessor language, but the semantics of how language constucts behave within the macro and how they compose with other langauge features is still completely up to the programmer and, in my experience, quite hard to get right.

I see macros used in places that I find really troubling. I was writing a toy GTK application in Rust earlier today and learned that you need to use these weird macros to get memory management to play nicely with Rust.
```rust
use glib::clone;

let window = Rc::new(ApplicationWindow::new(app));

# moving a weak reference to `window` into the closure
butten.connect_activate(clone!(@weak window => move |_| {
    window.close(&button);
}));
```

I really don't think introducing this metalanguage is a good idea at all. Also, how is this custom syntax supposed to be understood by the editor? Before `rust-analyzer` my editor (VSCode + RLS) would give up with the macro and I would have to guess my way out. Things are better now that we have `rust-analyzer` but I'm not even sure the Rust tooling is ever supposed to be able to make sense of this.

Bottom line (and I'm happy to be proven wrong): macros are an unsustainable convenience. They are never good enough to justify the readability/maintainability/tooling headaches.