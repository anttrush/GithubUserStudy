from neo4j.v1 import GraphDatabase
from functools import reduce

def countfn():
    uri = "bolt://10.1.1.4:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "buaaxzl"))
    # idlist = [] # 18944566

    with driver.session() as session:
        id = 1
        step = 100 # 500
        rounds = 189445 # 37889
        for r in range(rounds):
            with session.begin_transaction() as tx:
                idlist = [str(x) for x in range(id+r*step, id+(r+1)*step)]
                idlist = reduce(lambda x,y:x+','+y, map(str, idlist))
                records = tx.run("\
                start n=node(%s) \
                Match (n:User)-[r1:Report]->(:Issue) \
                with n, count(r1) as rin \
                set n.reportIssueNum = rin" % idlist)
                records = tx.run("\
                start n=node(%s) \
                Match (n:User)-[r2:TriggerEvent{action:'closed'}]->(:Issue) \
                with n, count(r2) as cin \
                set n.closeIssueNum = cin" % idlist)
            print("round %d, node %d" % (r, id+(r+1)*step))

countfn()