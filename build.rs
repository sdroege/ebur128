fn main() {
    let mut b = cc::Build::new();

    if cfg!(feature = "internal-tests") {
        b.define("TESTS", "1");
        b.compiler("clang");
        b.file("src/c/tests/interp.c");
        b.file("src/c/tests/true_peak.c");
        b.file("src/c/tests/history.c");
    }

    b.file("src/c/ebur128.c");
    b.compile("ebur128");
}
