from cutlass import cute


@cute.jit
def bymode_coalesce_example():
    """
    Demonstrates by-mode coalescing
    """
    layout = cute.make_layout( ((2, 4), (1, 6)), stride=((1, 2), (6, 2)))

    # target_profile 某个维度 传数字就表示需要coalesce, 传tuple()就表示不需要coalesce
    # Coalesce with mode-wise profile (1,1) = coalesce both modes
    # result = cute.coalesce(layout, target_profile=(1, 1))  # (8,6):(1,2)
    # result = cute.coalesce(layout, target_profile=(1, tuple()))  # (8,(1,6)):(1,(6,2))
    result = cute.coalesce(layout, target_profile=(tuple(), 1))  # ((2,4),6):((1,2),2)
    
    # Print results
    print(">>> Original: ", layout)
    print(">>> Coalesced Result: ", result)


bymode_coalesce_example()
