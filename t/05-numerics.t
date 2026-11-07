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
        like(`./test_runner $id $s`, qr/RET:0\b/, "$strategies[$s]: $name");
    }
}
for my $s (1 .. 3) {
    for my $id (39, 40) {
        my $storage = $id == 39 ? 'dense' : 'sparse';
        like(`./test_runner $id $s`, qr/RET:0\b/,
             "$strategies[$s]: $storage objective scale invariance");
    }
}
for my $case ([41, 'dense scaled inactive affine constraint'],
              [42, 'sparse scaled inactive affine constraint'],
              [43, 'dense scaled equality and active box bound'],
              [44, 'sparse scaled equality and active box bound'],
              [45, 'dense tight ADMM tolerance'], [46, 'sparse tight ADMM tolerance'],
              [48, 'dense affine iteration limit'], [49, 'sparse affine iteration limit']) {
    my ($id, $name) = @$case;
    like(`./test_runner $id 4`, qr/RET:0\b/, $name);
}

done_testing();
