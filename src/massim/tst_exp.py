import massim.experiment as EX
reload(EX)

exp = EX.Experiment("test")

sweep = EX.ParameterSweep(num_steps=5).add_sweep("Debug_1",
                                                 "test_val",
                                                 EX.LinSpace(7,2))

root = exp.set_start_stage(sweep)
root.link_stage(EX.DebugStage("DBG1")).\
    link_stage(EX.ReplicateStage(replicates=2)).\
    link_stage(EX.DebugStage("DBG2", test_val=5))
