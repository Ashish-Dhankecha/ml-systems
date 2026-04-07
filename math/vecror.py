def normalize(v):
    if not v:
        raise ValueError("Empty vector not allowed")

    if isinstance(v[0], list):
        if len(v[0]) == 1:
            return [x[0] for x in v], "col"
        elif len(v) == 1:
            return v[0], "row"
        else:
            raise ValueError("Invalid matrix shape for vector")

    return v, "flat"
def prepare_vectors(a, b):
    a, type_a = normalize(a)
    b, type_b = normalize(b)

    if len(a) != len(b):
        raise ValueError("Vectors must be same length")

    return a, b, type_a, type_b

def add(a, b):
    """
    Adds two vectors element-wise.

    Supports flat lists, row vectors, and column vectors.
    Returns a column vector only if both inputs are columns.

    Raises error if lengths do not match.
    """
    a, b, type_a, type_b = prepare_vectors(a, b)
    result = [x + y for x, y in zip(a, b)]

    if type_a == "col" and type_b == "col":
        return [[x] for x in result]

    return result


def sub(a, b):
    """
    Subtracts vector b from vector a.

    Works with flat, row, and column formats.
    Keeps column structure if both inputs are columns.
    """
    a, b, type_a, type_b = prepare_vectors(a, b)
    result = [x - y for x, y in zip(a, b)]

    if type_a == "col" and type_b == "col":
        return [[x] for x in result]

    return result


def dot(a, b):
    """
    Computes dot product of two vectors.

    Input can be flat, row, or column.
    Returns a single number.
    """
    a, b, _, _ = prepare_vectors(a, b)
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    """
    Computes Euclidean norm (magnitude) of a vector.
    """
    return dot(a, a) ** 0.5