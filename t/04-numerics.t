use strict;
use warnings;
use Test::More;

my @strategies = qw(CD PG AG SPG ADMM);
for my $s (0 .. $#strategies) {
    for my $case ([33, 'dense fixed coordinate'], [34, 'sparse fixed coordinate'],
                  [35, 'dense tiny positive diagonal'], [36, 'sparse tiny positive diagonal'],
                  [37, 'dense projected step cycle'], [38, 'sparse projected step cycle'],
                  [47, 'least-squares scale continuity']) {
        my ($id, $name) = @$case;
        like(`./t/00util/test_runner $id $s`, qr/RET:0\b/, "$strategies[$s]: $name");
    }
}
for my $s (0 .. $#strategies) {
    for my $id (39, 40) {
        my $storage = $id == 39 ? 'dense' : 'sparse';
        like(`./t/00util/test_runner $id $s`, qr/RET:0\b/,
             "$strategies[$s]: $storage objective scale invariance");
    }
}
for my $s (3, 4) {
    for my $case ([41, 'dense scaled inactive affine constraint'],
                  [42, 'sparse scaled inactive affine constraint'],
                  [43, 'dense scaled equality and active box bound'],
                  [44, 'sparse scaled equality and active box bound'],
                  [51, 'dense separate box/affine primal scaling'],
                  [52, 'sparse separate box/affine primal scaling'],
                  [53, 'dense multiscale pose+global affine QP'],
                  [54, 'sparse multiscale pose+global affine QP'],
                  [55, 'dense high-scale initial penalty seeding'],
                  [56, 'sparse high-scale initial penalty seeding'],
                  [57, 'dense multi-row affine ALM-SPG vs ADMM parity'],
                  [58, 'sparse multi-row affine ALM-SPG vs ADMM parity']) {
        my ($id, $name) = @$case;
        like(`./t/00util/test_runner $id $s`, qr/RET:0\b/,
             "$strategies[$s]: $name");
    }
}
for my $case ([45, 'dense tight ADMM tolerance'],
              [46, 'sparse tight ADMM tolerance'],
              [48, 'dense affine iteration limit'],
              [49, 'sparse affine iteration limit'],
              [59, 'dense unconstrained PCG fast-path'],
              [60, 'sparse unconstrained PCG fast-path']) {
    my ($id, $name) = @$case;
    like(`./t/00util/test_runner $id 4`, qr/RET:0\b/, $name);
}
for my $case ([61, 'SPG: dense non-convex [0,1] negative-curvature clique QP'],
              [62, 'SPG: sparse non-convex [0,1] negative-curvature clique QP']) {
    my ($id, $name) = @$case;
    like(`./t/00util/test_runner $id 3`, qr/RET:0\b/, $name);
}

done_testing();
