from megatron.core import parallel_state


def test_parallel_state():
    decoder_rank_generator = parallel_state.RankGenerator(
        tp=2,
        ep=1,
        dp=2,
        pp=4,
        cp=2,
        order="tp-cp-ep-dp-pp",
        rank_offset=0,
    )
    print("tp:", decoder_rank_generator.get_ranks("tp"))
    print("cp:", decoder_rank_generator.get_ranks("cp"))
    print("ep:", decoder_rank_generator.get_ranks("ep"))
    print("dp:", decoder_rank_generator.get_ranks("dp"))
    print("pp:", decoder_rank_generator.get_ranks("pp"))
    print("pp-tp:", decoder_rank_generator.get_ranks("pp-tp"))


if __name__ == "__main__":
    test_parallel_state()
