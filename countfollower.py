from neo4j.v1 import GraphDatabase
from functools import reduce

def countfn():
    uri = "bolt://10.1.1.4:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "buaaxzl"))
    # idlist = [] # 18944566

    with driver.session() as session:
        with session.begin_transaction() as tx:
            id = 16440001
            step = 5000 # 5000
            rounds = 500 # 3788
            for r in range(rounds):
                idlist = [str(x) for x in range(id+r*step, id+(r+1)*step)]
                idlist = reduce(lambda x,y:x+','+y, map(str, idlist))
                records = tx.run("\
                start n=node(%s) \
                Match (n:User)<-[:Follow]-(m:User) \
                with n, count(m) as fn \
                set n.followNum = fn \
                return n.userId, n.followNum" % idlist)
                print("round %d" % r)

countfn()