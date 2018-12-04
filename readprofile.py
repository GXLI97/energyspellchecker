import pstats
from pstats import SortKey
p = pstats.Stats('profile_output')
# p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

