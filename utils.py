import torch


def save_checkpoint(epoch, model, optimizer, filename):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    torch.save(state, filename)


def get_labels(num_parts, num_angles):
    res_labels = []
    co = 0
    for i in range(num_parts):
        line = list()
        for j in range(num_angles):
            line.append(co)
            co += 1
        res_labels.append(line)
    return res_labels  

    