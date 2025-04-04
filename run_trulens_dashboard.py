from trulens_eval import Tru

tru = Tru()

#TruSession().migrate_database(prior_prefix="")
TruSession.reset_database()

tru.run_dashboard()
