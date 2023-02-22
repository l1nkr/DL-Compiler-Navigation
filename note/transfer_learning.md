# 随手，未整理版

api: load_history()
gallery/how_to/tune_with_autotvm
        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
key_word: transfer
python/tvm/autotvm/tuner/
但是里面好像都没有实现，一会再找一下

python/tvm/driver/tvmc/autotuner.py
        # If transfer learning is being used, load the existing results
        if tuning_records and os.path.exists(tuning_records):
            logger.info("loading tuning records from %s", tuning_records)
            start_time = time.time()
            tuner_obj.load_history(autotvm.record.load_from_file(tuning_records))
            logging.info("loaded history in %.2f sec(s)", time.time() - start_time)

https://discuss.tvm.apache.org/t/autotvm-clarification/8766/4

当你使用XGBTuner的load_history时，XGBTuner中的cost model会根据加载的调优日志进行训练，以便更快地找到更好的配置； 否则它从随机配置开始，并使用它们的测量结果来训练新的成本模型。
谢谢，所以理论上我可以说如果保存的调整日志有足够的信息（或调整试验），那么我可以离线训练成本模型（我的意思是没有真正的硬件目标），因为从用户方面来说，他们不想要 在为同一硬件目标部署到生产环境之前调整每个模型/OP（时间成本）。
在这种情况下，直接保存您已调整的每个操作的最佳配置实际上更好。 否则，您的用户仍然需要使用经过训练的成本模型来调整操作以在硬件设备上进行一些试验。 您可以参考本演示文稿中的想法
https://discuss.tvm.apache.org/t/transfer-learning-doesnt-work-tuner-obj-load-history/5328/2


这里的迁移学习是在调优完一个模型之后，然后得到一个tmp.log。在调优第二个模型之前，先读入之前的tmp.log，用这个tmp.log提前训练cost model。
那么对于同一个模型的调优，并没有使用到迁移学习的技术。

