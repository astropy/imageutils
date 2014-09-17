import unittest
import laxtest
suite = unittest.TestLoader().loadTestsFromTestCase(laxtest.Test)
unittest.TextTestRunner(verbosity=2).run(suite)

