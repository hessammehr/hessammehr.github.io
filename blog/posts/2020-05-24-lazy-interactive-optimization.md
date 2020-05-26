# The case for lazy computation and interactive optimization
Since 2018, my colleague [Dario] and I have been working on a probabilistic model of chemical reactivity. In a nutshell, this model can take reactivity observations between a bunch of compounds and interpret them as manifestations of properties and their mutual reactivities. The [Markov chain Monte carlo] implementation of this model in [PyMC3] was quite fast to start but over time we have expanded the model and now with all the bells and whistles sampling can take close to a day.

Given these performance issues it was only natural to look at alternative implementations in my high performance language of choice, Julia. So, over the past year I have done maybe a dozen experiments, from a more or less equivalent high-level description in [Turing.jl], to encoding the log probability manually (on the CPU and the GPU) and using [DynamicHMC.jl] and friends to sample, to using the HMC implementation in [AdvancedHMC.jl] and hand coding the log probability + domain transformations. What has been surprising is that the performance gain hasn't been nearly as large as I imagined it would be. In fact, my fastest implementation using AdvancedHMC is still only half as fast as the straightforward PyMC3 implementation. There may be differences in the formulation of NUTS between the different packages, of course, so I don't think this should be taken as representative. Just that it got me thinking about performance.

Yesterday, I started thinking about PyMC3's backend, [Theano], and how it allows PyMC to be fast and expressive. Theano (and newer systems like TensorFlow) express computations as executable graphs of operations (_ops_ for short). In principle, evaluation of these graphs can entail any of the following:

1. Simply running an interpreted implementation of each _op_.
2. Native code generation for all or part of the ops before execution.
3. Graph optimization, e.g. removing redundant ops or rewriting sequences of ops them with faster equivalents.

Libraries like Theano and TensorFlow have a fairly limited scope (numerical code) but I believe that borrowing certain of the above elements can make a great DSL for high performance computing in other domains. Specifically:

1. Computations described as graphs of ops.
2. Op-graph transformations, themselves described using #1. These transformations can be applied to a certain op or to all ops in a given scope (_e.g._ children of a certain op, or ops matching a certain pattern).
3. Interpretive execution of the op graph.

Using this design, optimizations will be part of libraries that can be imported and applied to existing code _a la carte_ as opposed to hard-coded in the compiler/JIT. This model is somewhat similar to what Julia does, _i.e._ interpretation and JIT compilation of code based on inferred type, but goes beyond building everything around the type system. Moreover, it can be implemented as a library in Julia, Python, or any language with bindings to codegen backend being targetted (if any).

**Update (2020-05-26):** Relevant discussion on the [Julia discourse]. Forum users pointed out some of the promising developments in the area, e.g. [Mjolnir] and being able to [customize the compilation pipeline] through parameterized interpretation.

[AdvancedHMC.jl]: https://github.com/TuringLang/AdvancedHMC.jl
[customize the compilation pipeline]: https://github.com/JuliaLang/julia/pull/33955
[Dario]: https://twitter.com/DarioCaramelli
[DynamicHMC.jl]: https://github.com/tpapp/DynamicHMC.jl
[Julia discourse]: https://discourse.julialang.org/t/idea-scope-rather-than-type-centric-composable-optimizations
[Markov chain Monte carlo]: https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
[Mjolnir]: https://github.com/MikeInnes/Mjolnir.jl
[PyMC3]: https://docs.pymc.io
[Theano]: http://www.deeplearning.net/software/theano/
[Turing.jl]: https://turing.ml