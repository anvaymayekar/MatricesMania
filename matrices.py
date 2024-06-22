from typing import Any, Callable, Sequence, SupportsIndex
from copy import deepcopy

# author: @anvaymayekar

Number = int | float | complex
Matrix = list[list[Number]]
Call = Callable[..., Any]


class Matrices:
    def __init__(self) -> None:
        pass

    @staticmethod
    def display(matrix: Matrix) -> str:
        res: str = "\n"
        for row in matrix:
            res += f" ".join(map(lambda x: str(x), row))
            res += "\n"

        return res

    @staticmethod
    def balance(matrix: Matrix, head: Number = 0) -> bool:
        l: int = len(matrix)

        for row in range(l):
            q: int = len(matrix) - len(matrix[row])

            if q > 0:
                matrix[row].extend([head] * q)
            elif q < 0:
                del matrix[row][l : (l - q)]

    @staticmethod
    def checkpost(func: Call) -> Call:
        def wrapper(*args: Any, **kwargs: Any) -> Matrix | str:
            try:
                arguments: tuple[Any] = args + tuple(v for _, v in kwargs.items())
                scheme: bool = kwargs["scheme"] if "scheme" in kwargs.keys() else False
                for arg in arguments:
                    if type(arg) is list and type(arg[0]) is list:
                        Matrices.balance(arg)

                ref: Matrix | Number = func(*args, **kwargs)
                return Matrices.display(ref) if scheme and type(ref) == list else ref
            except Exception as e:
                raise "Failed to exercise the method!"

        return wrapper

    @staticmethod
    def __order(matrix: Matrix) -> int:
        return len(matrix)

    @staticmethod
    def __slice_matrix(
        matrix: Matrix,
        row: SupportsIndex,
        col: SupportsIndex,
    ) -> Matrix:
        new_matrix: Matrix = deepcopy(matrix)
        del new_matrix[row]
        l: int = len(new_matrix)

        for i in range(l):
            del new_matrix[i][col]

        return new_matrix

    @checkpost
    def transpose(
        self, matrix: Matrix, scheme: bool = False, *args, **kwargs
    ) -> Matrix:
        cpymatrix: Matrix = deepcopy(matrix)
        l: int = len(cpymatrix)
        for i in range(l):
            for j in range(l):
                cpymatrix[i][j] = matrix[j][i]

        return cpymatrix

    def __dual_multiplication(self, A: Matrix, B: Matrix) -> Matrix:
        product: Matrix = []
        l: int = len(A)
        m: int = len(B)

        if l != m:
            print(f"Incompatible order {l} & {m}")
            return

        for i in range(l):
            for j in range(l):
                temp: Number = 0
                for k in range(l):
                    temp += A[i][k] * B[k][j]
                if i == 0:
                    product.append([])
                product[i].append(temp)

        return product

    @checkpost
    def multipy(self, *args, power: int | None = None, **kwargs) -> Matrix | str:
        prev: Matrix = args[0]
        if power != None and len(args) == 1:
            args = list(args * power)
        a: int = len(args)

        for i in range(1, a):
            cur: Matrix = self.__dual_multiplication(prev, args[i])
            prev = cur

        return prev

    @checkpost
    def scalar_multiply(self, matrix: Matrix, factor: Number = 1) -> Matrix | str:
        l: int = len(matrix)
        cpymatrix: Matrix = deepcopy(matrix)
        for i in range(l):
            for j in range(l):
                cpymatrix[i][j] *= factor

        return cpymatrix

    @checkpost
    def generate_identity(
        self, order: int = 2, factor: Number = 1, *args, **kwargs
    ) -> Matrix:
        res: Matrix = [[] for _ in range(order)]

        for i in range(order):
            for j in range(order):

                if i == j:
                    res[i].append(factor * 1)

                else:
                    res[i].append(0)

        return res

    @checkpost
    def determinant(self, matrix: Matrix, *args, **kwargs) -> Call | Number:
        order: int = len(matrix)
        if order == 1 and len(matrix[0]) == 1:
            return matrix[0][0]
        if order == 2:
            # base case
            simple_det: Number = (
                matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            )
            return simple_det
        else:
            # cofactor expansion
            quant: Number = 0
            for j in range(order):
                sliced_matrix: Matrix = Matrices.__slice_matrix(matrix, 0, j)
                cofactor: Number = ((-1) ** j * matrix[0][j]) * self.determinant(
                    sliced_matrix
                )
                quant += cofactor

            return quant

    @checkpost
    def condense(
        self, array: Sequence[Number], order: int = 3, *args, **kwargs
    ) -> Matrix | str:

        array: list[Number] = list(array)
        res: Matrix = []
        i: int = 0
        for _ in range(order):
            res.append(array[i:order])
            i = order
            order += order
        self.balance(res)
        return res

    @checkpost
    def cofactor(self, matrix: Matrix, **kwargs) -> Matrix | str:
        l: int = len(matrix)
        cofactor_matrix: Matrix = [[] for _ in range(l)]

        for row in range(l):
            for col in range(l):

                sliced: Matrix = self.__slice_matrix(matrix, row, col)
                det: Number = self.determinant(sliced)

                cofactor_matrix[row].append(((-1) ** (row + col)) * det)

        return cofactor_matrix

    @checkpost
    def adjoint(self, matrix: Matrix, scheme=False, **kwargs) -> Matrix | str:
        cofactor_matrix: Matrix = self.cofactor(matrix, scheme=False)
        return self.transpose(cofactor_matrix, scheme=scheme)

    @checkpost
    def inverse(self, matrix: Matrix, **kwargs) -> Matrix | str:
        det: Number = self.determinant(matrix)
        adj: Matrix = self.adjoint(matrix)
        res: Matrix = self.scalar_multiply(adj, factor=det)

        return res


a = [
    [1, 2, 6, 8],
    [4, 6, 9, 8],
    [7, 9, 7, 9],
    [9, 9, 6, 9],
]

b = [
    [1, 7, 8],
    [8, 1, 5],
    [9, 0, 6],
]

# deployment

m = Matrices()
print(m.generate_identity())
print(
    m.determinant(
        [
            [1, 2],
            [4, 5],
        ]
    )
)

print(m.display(a))
print(m.transpose(matrix=a, scheme=True))
j = m.generate_identity(4, 6)
print(m.scalar_multiply(a, 2))
print(m.multipy(b, power=5, scheme=False))
print(
    m.cofactor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
)


print(m.cofactor(m.generate_identity(3, scheme=False)))
print(m.generate_identity(3, scheme=False))


print(m.cofactor([[1, 2], [3, 4]], scheme=True))
print(m.determinant(a))
