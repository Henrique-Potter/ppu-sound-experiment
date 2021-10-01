from up_framework.up_engine_interface import IUPEngine


class UPEngine:

    def run_up_engine(self, up_instance: IUPEngine):

        up_instance.load_data()
        up_instance.pre_process()
        up_instance.run_model()
        up_instance.process_results()

