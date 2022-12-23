import numpy as np
import torch
import paddle

finetune = 'checkpoint-1599.pth'
checkpoint = torch.load(finetune, map_location='cpu')
checkpoint_model = checkpoint['model']    
    
# convert torch weight to paddle weight
for item0 in list(checkpoint_model.items()):
    name = item0[0]
    np0 = item0[1].numpy()

    if name == 'head.0.num_batches_tracked':
        del checkpoint_model[name]
        continue

    if name == 'head.0.running_mean':
        del checkpoint_model[name]
        name = 'head.0._mean'
    if name == 'head.0.running_var':
        del checkpoint_model[name]
        name = 'head.0._variance'

    need_transpose = len(np0.shape) == 2
    if need_transpose:
        np0 = np0.transpose()
        print(name, np0.shape, 'transposed')
    else:
        print(name, np0.shape)
        # pass
    checkpoint_model[name] = np0
    
to_save = {
    'model': checkpoint_model,
}
paddle.save(to_save, "checkpoint-0.pdparams")
