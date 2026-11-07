use strict;
use warnings;
use Test::More;
use Cwd qw(getcwd);
use File::Copy qw(copy);
use File::Path qw(make_path);
use File::Temp qw(tempdir tempfile);
use Text::ParseWords qw(shellwords);

my $root = getcwd();
my $tmp = tempdir(CLEANUP => 1);

sub write_file {
    my ($path, $text) = @_;
    open my $file, '>', $path or die "$path: $!";
    print {$file} $text;
    close $file or die "$path: $!";
}

sub run_command {
    my @command = @_;
    my ($log, $path) = tempfile(DIR => $tmp);
    my $pid = fork();
    die "fork: $!" unless defined $pid;
    if (!$pid) {
        open STDOUT, '>&', $log or die "stdout: $!";
        open STDERR, '>&', $log or die "stderr: $!";
        exec { $command[0] } @command;
        die "exec $command[0]: $!";
    }
    waitpid($pid, 0);
    my $status = $?;
    seek $log, 0, 0;
    local $/;
    my $output = <$log> // '';
    close $log;
    unlink $path;
    return ($status, $output);
}

# Exercise the real dependency graph without requiring a second native or
# cross toolchain. These tools record each build and emit GCC-style .d files.
my $fixture = "$tmp/build";
make_path("$fixture/t", "$fixture/bin");
copy('Makefile', "$fixture/Makefile") or die "copy Makefile: $!";
for my $source (qw(nanoqsp.c nanoqsp_blas.c example.c solve_cli.c
                   benchmark_sparse.c t/test_runner.c)) {
    write_file("$fixture/$source", "/* build graph fixture */\n");
}
write_file("$fixture/bin/compiler", <<'COMPILER');
#!/usr/bin/env perl
use strict;
use warnings;
my ($out, $source);
for my $i (0 .. $#ARGV) {
    $out = $ARGV[$i + 1] if $ARGV[$i] eq '-o';
    $source = $ARGV[$i] if $ARGV[$i] =~ /\.c$/;
}
die 'missing compiler output/source' unless defined $out && defined $source;
open my $file, '>', $out or die "$out: $!";
print {$file} join(' ', @ARGV), "\n";
close $file;
(my $dep = $out) =~ s/\.o$//;
open my $deps, '>', "$dep.d" or die "$dep.d: $!";
print {$deps} "$out: $source\n";
COMPILER
write_file("$fixture/bin/archiver", <<'ARCHIVER');
#!/usr/bin/env perl
use strict;
use warnings;
shift @ARGV;
my $out = shift @ARGV;
open my $file, '>', $out or die "$out: $!";
for my $object (@ARGV) {
    open my $input, '<', $object or die "$object: $!";
    print {$file} <$input>;
}
ARCHIVER
write_file("$fixture/bin/remove", <<'REMOVE');
#!/usr/bin/env perl
use strict;
use warnings;
# A short delay makes concurrently scheduled `check` reliably observe that
# cleanup has not finished, instead of depending on scheduler timing.
unlink '.clean-complete';
select undef, undef, undef, 0.1;
unlink @ARGV;
open my $done, '>', '.clean-complete' or die $!;
REMOVE
write_file("$fixture/bin/prove", <<'PROVE');
#!/usr/bin/env perl
use strict;
use warnings;
die 'check ran before cleanup finished' unless -e '.clean-complete';
for my $path (qw(nanoqsp.o nanoqsp_blas.o libnanoqsp.a example test_runner
                 solve_cli benchmark_sparse)) {
    open my $file, '<', $path or die "$path: $!";
    local $/;
    my $flags = <$file>;
    die "$path was not rebuilt with sanitizers" unless $flags =~ /-fsanitize=address,undefined/;
}
PROVE
chmod 0755, map { "$fixture/bin/$_" } qw(compiler archiver remove prove);

{
    # Do not inherit outer `make check` command-line variables or its jobserver.
    local %ENV = %ENV;
    delete @ENV{qw(MAKEFLAGS MFLAGS MAKELEVEL MAKEOVERRIDES)};
    $ENV{PATH} = "$fixture/bin:$ENV{PATH}";
    chdir $fixture or die "chdir $fixture: $!";
    my @make = (shellwords($ENV{NANOQSP_MAKE} // 'make'), '--no-print-directory',
                "CC=$fixture/bin/compiler", "AR=$fixture/bin/archiver");
    my ($status, $output) = run_command(@make, 'CLASSIC=1');
    is($status, 0, 'initial default build succeeds') or diag $output;
    ($status, $output) = run_command(@make);
    is($status, 0, 'default incremental backend change succeeds') or diag $output;
    ($status, $output) = run_command(@make, '-q', 'all');
    is($status, 0, 'default build updates every backend object and linked target') or diag $output;
    ($status, $output) = run_command(@make, '-j8', "RM=$fixture/bin/remove", 'check-asan');
    is($status, 0, 'parallel sanitizer check waits for cleanup and rebuilds all targets') or diag $output;
    chdir $root or die "chdir $root: $!";
}

# Using the selected C compiler's C++ frontend preserves its target and sysroot.
# Cross CI may intentionally install only the C frontend; native CI has g++.
my @compiler = shellwords($ENV{NANOQSP_CC} // $ENV{CC} // 'cc');
my @cppflags = shellwords($ENV{NANOQSP_CPPFLAGS} // $ENV{CPPFLAGS} // '');
my @cflags = grep { !/^-std=/ } shellwords($ENV{NANOQSP_CFLAGS} // '');
my @ldflags = shellwords($ENV{NANOQSP_LDFLAGS} // '-lm');
write_file("$tmp/probe.cpp", "int main() { return 0; }\n");
my ($probe_status, $probe_output) = run_command(@compiler, @cppflags, @cflags,
    '-x', 'c++', '-std=c++11', '-fsyntax-only', "$tmp/probe.cpp", '-o', "$tmp/probe");
SKIP: {
    skip 'selected cross compiler has no C++ frontend', 2
        if $probe_status && $probe_output =~ /cc1plus.*(?:No such file|not found)/s;
    write_file("$tmp/header.cpp", <<'CPP');
#include "nanoqsp.h"
int main() {
    double D[] = {2.0}, d[] = {4.0}, x[] = {0.0};
    NanoqspConfig config = {};
    config.strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
    int status = nanoqsp_solve_box(1, D, d, 0, 0, x, &config);
    double prediction = nanoqsp_predict(1, x, d);
    return status < 0 || prediction < 7.999999 || prediction > 8.000001;
}
CPP
    my ($status, $output) = run_command(@compiler, @cppflags, @cflags,
        '-x', 'c++', '-std=c++11', "-I$root", "$tmp/header.cpp", '-x', 'none',
        "$root/libnanoqsp.a", @ldflags, '-o', "$tmp/header");
    is($status, 0, 'public header compiles in C++ and links to the C library') or diag $output;
    ($status, $output) = $status ? ($status, 'C++ compilation failed') : run_command("$tmp/header");
    is($status, 0, 'C++ caller solves and predicts through the public API') or diag $output;
}

done_testing();
