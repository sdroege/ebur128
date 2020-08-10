fn main() {
    #[cfg(feature = "c-tests")]
    {
        let mut b = cc::Build::new();

        b.define("TESTS", "1");
        b.compiler("clang");
        b.file("tests/c/interp.c");
        b.file("tests/c/true_peak.c");
        b.file("tests/c/history.c");
        b.file("tests/c/filter.c");
        b.file("tests/c/calc_gating_block.c");
        b.compile("ebur128");
    }
}
