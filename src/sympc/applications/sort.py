def sort_mpctensor_list(input_list, ascending=True):
    if len(input_list) > 1:
        mid = len(input_list) // 2
        left = input_list[:mid]
        right = input_list[mid:]

        # Recursive call on each half
        sort_mpctensor_list(left)
        sort_mpctensor_list(right)

        # Two iterators for traversing the two halves
        i = 0
        j = 0

        # Iterator for the main list
        k = 0

        while i < len(left) and j < len(right):
            if (left[i] < right[j]).reconstruct():
                # The value from the left half has been used
                input_list[k] = left[i]
                # Move the iterator forward
                i += 1
            else:
                input_list[k] = right[j]
                j += 1
            # Move to the next slot
            k += 1

        # For all the remaining values
        while i < len(left):
            input_list[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            input_list[k] = right[j]
            j += 1
            k += 1

    if not ascending:

        return input_list[::-1]

    return input_list
