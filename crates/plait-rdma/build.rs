fn main() {
    // rdma-io-sys links the verbs symbols but does not publish the system
    // library search path. pkg-config finds both the normal distro location and
    // custom prefixes such as /usr/local.
    pkg_config::Config::new()
        .atleast_version("1.14")
        .probe("libibverbs")
        .expect("libibverbs development files are required");
    pkg_config::Config::new()
        .atleast_version("1.3")
        .probe("librdmacm")
        .expect("librdmacm development files are required");
}
