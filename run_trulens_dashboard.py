from trulens_eval import Tru
from trulens.core import TruSession

#database_file="default.sqlite"

tru = Tru()

#session = TruSession()
#session.migrate_database(prior_prefix="")

#TruSession().migrate_database(prior_prefix="")
#tru .reset_database()

tru.run_dashboard()
