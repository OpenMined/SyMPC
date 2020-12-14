from sympc.tensor import ShareTensor


def count_wraps(share_list: List[ShareTensor]) -> int
    """ Count the number of overflows and underflows that would happen if we
    reconstruct the original value

    This function is taken from the CrypTen project.
    CrypTen repository: https://github.com/facebookresearch/CrypTen

    Args:
        share_list (List[ShareTensor]): List of the shares

    Returns:
        The number of wraparounds
    """

    res = torch.zeros(size=share_list[0].size(), dtype=torch.long)
    prev_share = share_list[0]
    for cur_share in share_list[1:]:
        next_share = cur_share + prev_share
        result += ((prev_share < 0) & (cur_share > 0) & (next_share > 0)).long()
        result -= ((prev_share > 0) & (cur_share > 0) & (next_share < 0)).long()
        prev_share = next_share

    return result
