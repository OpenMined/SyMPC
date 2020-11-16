from .. import Beaver

class SPDZ:
    @staticmethod
    def mul_master(x, y):

        """
        [c] = [a * b]
        [eps] = [x] - [a]
        [delta] = [y] - [b]

        Open eps and delta
        [result] = [c] + eps * [b] + delta * [a] + eps * delta
        """
        a_sh, b_sh, c_sh = Beaver.build_triples("mul", x, y)
        eps = x - a_sh
        delta = y - b_sh
        import pdb; pdb.set_trace()

        eps_plaintext = eps.reconstruct()
        delta_plaintext = delta.reconstruct()
        print("EPS", eps_plaintext)
        print("DELTA", delta_plaintext)

        res = eps_plaintext * b_sh
        print((res + 0).reconstruct())
        res = res + delta_plaintext * a_sh
        print((res + 0).reconstruct())
        res = res + eps_plaintext * delta_plaintext
        print((res + 0).reconstruct())
        res = res + c_sh
        print((res + 0).reconstruct())

        return res
