from trulens_eval import Tru

tru = Tru(database_file="../default.sqlite")

session = TruSession()
session.migrate_database(prior_prefix="")

#TruSession().migrate_database(prior_prefix="")
#tru .reset_database()

tru.run_dashboard()
