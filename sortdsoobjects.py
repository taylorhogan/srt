# for all dso objects
#   remove those that have been imaged, done
#   how many hours will it be visible > some altitude
#https://astroplan.readthedocs.io/en/stable/

from astroplan import Observer
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget

subaru = Observer.at_site('subaru')
my_observatory = Observer.at_site()

