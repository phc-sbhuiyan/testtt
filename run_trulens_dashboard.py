from trulens_eval import Tru

tru = Tru()

#TruSession().migrate_database(prior_prefix="")
tru .reset_database()

tru.run_dashboard()
