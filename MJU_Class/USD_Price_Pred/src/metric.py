# metric.py
# 원하는 metric 함수를 추가하면 됨. 단, mae는 반드시 남겨둘 것

import torch

def rmse(output, target):
    with torch.no_grad():
        return torch.sqrt(torch.mean((output - target) ** 2)).item()
    

def mae(output, target):
    with torch.no_grad():
        return torch.mean(torch.abs(output - target)).item()


def mape(output, target):
    with torch.no_grad():
        return torch.mean(torch.abs((output - target) / target)).item()
    

if __name__ == '__main__':
    import torch
    
    output = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    target = torch.tensor([2, 2, 3, 4, 7], dtype=torch.float32)
    
    print(f'RMSE : {rmse(output, target):.4f}')
    print(f'MAE  : {mae(output, target):.4f}')
    print(f'MAPE : {mape(output, target) * 100:.2f} %')
