fn main() {
    cc::Build::new().file("src/c/ebur128.c").compile("ebur128");
}
