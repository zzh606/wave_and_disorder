import sys
from XY_model import XYSystem

xy_system_1 = XYSystem(temperature=0.5, width=4)
xy_system_1.show()
print('Energy per spin:%.3f' % xy_system_1.energy)
