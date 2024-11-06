def test_monotonize():
    import numpy as np
    from xarray_behave.io.samplestamps.utils import monotonize, interpolator, ismonotonous

    inc_strict = np.array([0, 1, 2, 3])
    inc_nonstrict = np.array([0, 1, 2, 2])
    dec_strict = inc_strict[::-1]
    dec_nonstrict = inc_nonstrict[::-1]
    assert ismonotonous(inc_strict, direction="increasing", strict=True) == True
    assert ismonotonous(inc_strict, direction="increasing", strict=False) == True
    assert ismonotonous(inc_nonstrict, direction="increasing", strict=True) == False
    assert ismonotonous(inc_nonstrict, direction="decreasing", strict=False) == False
    assert ismonotonous(dec_strict, direction="decreasing", strict=True) == True
    assert ismonotonous(dec_nonstrict, direction="decreasing", strict=True) == False
    assert ismonotonous(np.array([1]), direction="increasing", strict=True) == True
    assert ismonotonous(np.array([1]), direction="increasing", strict=False) == True

    x = np.array([0, 1, 2, 2, 1])
    print(f"montonize {x}")
    print(f"  strict, inc: {monotonize(x)}")
    assert np.all(monotonize(x) == [0, 1, 2])
    print(f"  strict, dec: {monotonize(x, direction='decreasing')}")
    assert np.all(monotonize(x, direction="decreasing") == [0])
    print(f"  nonstrict, in: {monotonize(x, strict=False)}")
    assert np.all(monotonize(x, strict=False) == [0, 1, 2, 2])

    x = np.array([2, 1, 0, 0, 1])
    print(f"montonize {x}")
    print(f"  strict, inc: {monotonize(x)}")
    assert np.all(monotonize(x) == [2])
    print(f"  strict, dec: {monotonize(x, direction='decreasing')}")
    assert np.all(monotonize(x, direction="decreasing") == [2, 1, 0])
    print(f"  nonstrict, dec: {monotonize(x, strict=False, direction='decreasing')}")
    assert np.all(monotonize(x, strict=False, direction="decreasing") == [2, 1, 0, 0])
