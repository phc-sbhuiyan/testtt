from trulens_eval import Tru

tru = Tru(database_file="../default.sqlite")

#TruSession().migrate_database()
#TruSession().migrate_database(prior_prefix="")
#tru .reset_database()

tru.run_dashboard()
