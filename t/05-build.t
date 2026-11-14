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

# Exercise the real dependency graph without requiring a second toolchain.
my $fixture = "$tmp/build";
make_path("$fixture/include", "$fixture/lib", "$fixture/examples/svm",
          "$fixture/t/00util", "$fixture/bin");
copy('Makefile', "$fixture/Makefile") or die "copy Makefile: $!";
for my $hdr (qw(include/nanoqsp.h include/qsp_arena.h include/qsp_blas.h
                include/qsp_internal.h examples/svm/svm.h)) {
    write_file("$fixture/$hdr", "/* header fixture */\n");
}
for my $source (qw(lib/nanoqsp.c lib/qsp_blas.c lib/qsp_box.c lib/qsp_admm.c
                   lib/qsp_alm.c examples/least_squares.c examples/svm/svm.c
                   examples/svm/svm_train.c examples/svm/svm_predict.c
                   t/00util/test_runner.c benchmark.c)) {
    write_file("$fixture/$source", "/* source fixture */\n");
}
write_file("$fixture/bin/compiler", <<'COMPILER');
#!/usr/bin/env perl
use strict;
use warnings;
my ($out, $c_src, $o_src);
for my $i (0 .. $#ARGV) {
    $out = $ARGV[$i + 1] if $ARGV[$i] eq '-o';
    $c_src = $ARGV[$i] if $ARGV[$i] =~ /\.c$/;
    $o_src = $ARGV[$i] if !defined($o_src) && $ARGV[$i] =~ /\.o$/;
}
die 'missing compiler output' unless defined $out;
open my $file, '>', $out or die "$out: $!";
if (defined $c_src) {
    print {$file} join(' ', @ARGV), "\n";
    (my $dep = $out) =~ s/\.o$//;
    open my $deps, '>', "$dep.d" or die "$dep.d: $!";
    print {$deps} "$out: $c_src include/nanoqsp.h\n";
    close $deps;
} else {
    print {$file} join(' ', @ARGV), "\n";
    for my $arg (@ARGV) {
        next unless $arg =~ /\.(?:o|a)$/ && -e $arg;
        open my $in, '<', $arg or die "$arg: $!";
        print {$file} <$in>;
        close $in;
    }
}
close $file;
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
for my $path (qw(lib/nanoqsp.o lib/qsp_blas.o lib/qsp_box.o lib/qsp_admm.o
                 lib/qsp_alm.o libnanoqsp.a examples/least_squares
                 examples/svm/svm.o examples/svm/svm_train
                 examples/svm/svm_predict t/00util/test_runner benchmark)) {
    open my $file, '<', $path or die "$path: $!";
    local $/;
    my $flags = <$file>;
    die "$path was not rebuilt with sanitizers"
        unless $flags =~ /-fsanitize=address,undefined/;
}
PROVE
chmod 0755, map { "$fixture/bin/$_" } qw(compiler archiver remove prove);

{
    local %ENV = %ENV;
    delete @ENV{qw(MAKEFLAGS MFLAGS MAKELEVEL MAKEOVERRIDES)};
    $ENV{PATH} = "$fixture/bin:$ENV{PATH}";
    chdir $fixture or die "chdir $fixture: $!";
    my @make = (shellwords($ENV{NANOQSP_MAKE} // 'make'), '--no-print-directory',
                "CC=$fixture/bin/compiler", "AR=$fixture/bin/archiver");
    my ($status, $output) = run_command(@make);
    is($status, 0, 'initial default build succeeds') or diag $output;
    ($status, $output) = run_command(@make, '-q', 'all');
    is($status, 0, 'incremental build is up to date when inputs are unchanged') or diag $output;
    my $future = time() + 2;
    utime $future, $future, 'include/nanoqsp.h' or die "utime: $!";
    ($status, $output) = run_command(@make, '-q', 'all');
    isnt($status, 0, 'touching public header marks dependent targets dirty');
    ($status, $output) = run_command(@make);
    is($status, 0, 'incremental header rebuild succeeds') or diag $output;
    ($status, $output) = run_command(@make, '-j8', "RM=$fixture/bin/remove", 'check-asan');
    is($status, 0, 'parallel sanitizer check waits for cleanup and rebuilds all targets') or diag $output;
    chdir $root or die "chdir $root: $!";
}

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
    return status < 0 || x[0] < 1.999999 || x[0] > 2.000001;
}
CPP
    my ($status, $output) = run_command(@compiler, @cppflags, @cflags,
        '-x', 'c++', '-std=c++11', "-I$root/include", "$tmp/header.cpp", '-x', 'none',
        "$root/libnanoqsp.a", @ldflags, '-o', "$tmp/header");
    is($status, 0, 'public header compiles in C++ and links to the C library') or diag $output;
    ($status, $output) = $status ? ($status, 'C++ compilation failed') : run_command("$tmp/header");
    is($status, 0, 'C++ caller solves QP through the public API') or diag $output;
}

done_testing();
