import torch

def gpu_usage():
    try:
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    except:
        print('Cuda GPU was not detected')

class AverageMeter():
    def __init__(self, metric_name=''):
        self.total_sum = 0
        self.total_count = 0
        self.metric_name = metric_name

    def update(self, sum_values, count):
        self.total_sum += sum_values
        self.total_count += count

    def get_avg(self):
        return self.total_sum / self.total_count if self.total_count > 0 else 0

    def get_total(self):
        return self.total_sum

    def reset(self):
        self.total_sum = 0
        self.total_count = 0

    def __repr__(self):
        print(f'Average of metric: {self.metric_name} = {self.get_avg()}')