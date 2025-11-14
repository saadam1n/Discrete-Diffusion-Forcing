import torch

def remask_duplicates(x0, mask_token_id=0):
    # is each element equal to the next?

    same_contig = (x0[:, 1:] == x0[:, :-1])

    same_n = torch.cat(
        [
            same_contig,
            torch.zeros_like(x0[:, -1:])   
        ], dim=1
    )
    # is each element equal to the prev?
    same_p = torch.cat(
        [
            torch.zeros_like(x0[:, -1:]),   
            same_contig
        ], dim=1
    )

    same = torch.logical_or(same_n, same_p)

    print(x0[:, 1:])
    print(x0[:, :-1])
    print(same_contig)

    x0 = torch.where(same, mask_token_id, x0)
    return x0


arr = [[1, 1, 2, 7, 3, 7, 7, 9, 1, 1, 1, 1, 6, 8, 8, 8, 8, 8]]
arr = [[1, 1, 2, 3, 3]]
x = torch.LongTensor(arr)

print(x)
x = remask_duplicates(x)
print(x)