
import numpy as np
import random
import math
from collections import Counter
import copy
def get_plane(A, B, C):
    n = np.cross(B - A , C - A)
    d = -(n[0] * A[0] + n[1] * A[1] + n[2] * A[2])
    return (n, d)

def length(v):
  return math.sqrt(dotprod(v, v))
def dotprod(a,b):
    res = 0
    for i in range(3):
        res += a[i]*b[i]
    return res
def mirror_point(a, b, c, d, point):
    x1, y1, z1 = point
    k =(-a * x1-b * y1-c * z1-d)/(a * a + b * b + c * c)
    x2 = a * k + x1
    y2 = b * k + y1
    z2 = c * k + z1
    x3 = 2 * x2-x1
    y3 = 2 * y2-y1
    z3 = 2 * z2-z1
    return x3, y3, z3
# defining tetrahedrons
class Tetra:
    
    def __init__(self, sA, sB, sC, sD):
        self.A = np.array(sA)
        self.B = np.array(sB)
        self.C = np.array(sC)
        self.D = np.array(sD)
        
    def center(self):
        return ((self.A[0] + self.B[0] + self.C[0] + self.D[0])/4, 
        (self.A[1] + self.B[1] + self.C[1] + self.D[1])/4, 
        (self.A[2] + self.B[2] + self.C[2] + self.D[2])/4)
    
    def get_plot_points(self):
        points = []
        center = self.center()
        for vertex in [self.A, self.B, self.C, self.D]:
            points.append(vertex)
        return points
    
    def reflect(self, Vertex):
        
        if Vertex == "A":
            plane = get_plane(self.B, self.C, self.D)
            new_vertex = mirror_point(*plane[0], plane[1], self.A)
            vertices = (new_vertex, self.B, self.C, self.D)
        elif Vertex == "B":
            plane = get_plane(self.A, self.C, self.D)
            new_vertex = mirror_point(*plane[0], plane[1], self.B)
            vertices = (self.A, new_vertex, self.C, self.D)
        elif Vertex == "C":
            plane = get_plane(self.A, self.B, self.D)
            new_vertex = mirror_point(*plane[0], plane[1], self.C)
            vertices = (self.A, self.B, new_vertex, self.D)
        elif Vertex == "D":
            plane = get_plane(self.A, self.B, self.C)
            new_vertex = mirror_point(*plane[0], plane[1], self.D)
            vertices = (self.A, self.B, self.C, new_vertex)
            
        return Tetra(*vertices)
    
    def center_clash(self, seq, center_points = [], tetra_points = [], counter = 1):

        if self.center() in center_points:
             return False
        
        center_points.append(self.center())
        tetra_points.append(self.get_plot_points())
        if len(seq) == 1:
            reflection = self.reflect(seq)

            #print(reflection.center())

            if reflection.center() in center_points:
                if reflection.center() != center_points[0]:
                    return False
            
            return reflection, center_points + [reflection.center()], tetra_points + [reflection.get_plot_points()],counter+1
        else:
            #print(self.center())
            
          
            return self.reflect(seq[0]).center_clash(seq[1:], center_points, tetra_points, counter+1)
        

          
    def reflect_seq(self, seq, center_points = [], tetra_points = [], counter = 1):
        '''reflect the tetrahedron based on the give sequence'''
        if self.center() in center_points:
             return None,f"Crossing has happend.valid before index of {counter}",None,None
        
        center_points.append(self.center())
        tetra_points.append(self.get_plot_points())
        if len(seq) == 1:
            reflection = self.reflect(seq)
            #print(reflection.center())

            if reflection.center() in center_points:
                if reflection.center() != center_points[0]:
                    return None,f"Crossing has happend at index of {counter}",None
            
            return reflection, center_points + [reflection.center()], tetra_points + [reflection.get_plot_points()],counter+1
        else:
            #print(self.center())
            
          
            return self.reflect(seq[0]).reflect_seq(seq[1:], center_points, tetra_points, counter+1)

    def does_it_close(self, seq, center_points = [], tetra_points = [], counter = 1):
        '''checks if the knot closes'''
        if self.center() in center_points:
             return False
        
        center_points.append(self.center())
        tetra_points.append(self.get_plot_points())
        if len(seq) == 1:
            reflection = self.reflect(seq)

            #print(reflection.center())

            if reflection.center() in center_points:
                if reflection.center() == center_points[0]:
                    return True
                else:
                    return False
                
            
            return reflection, center_points + [reflection.center()], tetra_points + [reflection.get_plot_points()],counter+1
        else:
            #print(self.center())
            
          
            return self.reflect(seq[0]).does_it_close(seq[1:], center_points, tetra_points, counter+1)
        
    def get_new_center():
        pass
 
    def convert_back(self,seq):
        '''converts from A,B,C,D to u,d,r,l'''
        reflection = self
        centers = []
        res = ""
        for word in seq:
            reflection = reflection.reflect(word)
            centers.append(reflection.center())
         
            if word == "D":
                c_cur = centers[-1]
                c_prev = centers[-2]
                diff = eval_diff(c_cur,c_prev)
             
                if diff[0] == 0.5:
                    res += "f"
                elif diff[0] == -0.5:
                    res += "b"
                elif diff[1] == 0.5:
                    res += "r"
                elif diff[1] == -0.5:
                    res += "l"
                elif diff[2] == 0.5:
                    res += "u"
                elif diff[2] == -0.5:
                    res += "d"
        return res

    
    

    def get_all_transition(self,seq):
        '''List of all the transitions of the center of the tetrahedron'''
        res = []
        centers = []
        centers.append(self.center())
        for letter in seq:
            self = self.reflect(letter)
           
            centers.append(self.center())
            if len(centers)>1:
                c_cur = centers[-1]
                c_prev = centers[-2]
                diff = eval_diff(c_cur,c_prev)
                if diff not in res:
                    res.append(diff)
            
        return res
    
    def get_all_transition_now(self,seq):
        res = []
        centers = []
        centers.append(self.center())
        for letter in seq:
            self = self.reflect(letter)
           
            centers.append(self.center())
            if len(centers)>1:
                c_cur = centers[-1]
                c_prev = centers[-2]
                diff = eval_diff(c_cur,c_prev)
                
                res.append(diff)
            
        return res
                

def get_all_theta(transit):
    '''Return the all the angle between the centers of the tetrahedrons'''
    res = []
    for i in range(len(transit)):
        for j in range(i):
            prod = dotprod(transit[i],transit[j])
            cos = prod/(length(transit[i])*length(transit[j]))
            theta = np.arccos(cos)
            order = theta/(2*np.pi)*360


            if order not in res:
                res.append(order)
    return res           
        
    

    #rules:  aa=1, bb=1, cc= 1, dd = 1  ab=ba, ad = da, BD = DB, ACA = CAC, cdcd = dcDC


def generate_seq():
    '''generate all different approved sequences of knot. Uses breadth first search'''
    #t = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))
    #t = Tetra(np.array((0,-1,1)), np.array((0,1, 1)), np.array((-1, 0, 0))  ,np.array((-1,0,2)))
    t = Tetra(np.array((1,0,-1)), np.array((-1,0, 0)), np.array((1, -1, -1))  ,np.array((0,0,0)))
    moves = ["A","B","C","D"]
    stack = ["A","B","C","D"]

    res_close=[]
  
    file_not_close = open("ordereighDFS.txt", "w")
    while len(stack)>0:
        cur = stack.pop(0)
        order_three_cur = cur*2
        if t.center_clash(cur,[]) == False:
            continue
 
        #if len(cur)%2 == 0:
        if t.does_it_close(order_three_cur,[]) == True:      
            res_close.append(order_three_cur)     
            file_not_close.write(order_three_cur+"\n")
            print(order_three_cur)
        
            
        for move in moves:
            if len(cur+move)<=30:
                stack.append(cur+move)
    file_not_close.close()

    return 

def generate_seq_rand():
    '''generate all different approved sequences of knot. Uses breadth first search but add the points randomly'''
    t = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))
    moves = ["A","B","C","D"]
    stack = ["A","B","C","D"]

    res_close=[]
  
    file_not_close = open("orderthreeDFSrand.txt", "w")
    while len(stack)>0:
        cur = stack.pop()
        order_three_cur = cur*3
        if t.center_clash(cur,[]) == False:
            continue
 
        if len(cur)%2 == 0:
            if t.does_it_close(order_three_cur,[]) == True:      
                res_close.append(order_three_cur)     
                file_not_close.write(order_three_cur+"\n")
                print(order_three_cur)
        
        ran = random.randint(0,len(moves)-1)   
        for i in range(len(moves)):
            if len(cur)+1<=12:
                stack.append(cur+moves[(ran+i)%4])
    file_not_close.close()

    return 
class words:
    def __init__(self,word):
        self.word = word
    def counting(self):
        num_A = self.word.count("A")
        num_B = self.word.count("B")
        num_C = self.word.count("C")
        num_D = self.word.count("D")
        return num_A,num_B,num_C,num_D
    def stdev(self):
        avg = (self.counting()[0]+self.counting()[1]+self.counting()[2]+self.counting()[3])/4
        res = 0
        for i in range(4):
            res += (self.counting()[i]-avg)**2
        return (res/4)**0.5
    def __lt__(self,other):
        return self.stdev()<other.stdev()
  
def generate_seq_order():
    '''generate an ordered sequence. for instance "ABABAB" is a sequence of order 3 with AB repeatin three times'''
    t = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))
    moves = ["A","B","C","D"]
    stack = ["A","B","C","D"]

    res_close=[]
    
    file_not_close = open("orderthreeordered.txt", "w")
    while len(stack)>0:
        
        cur = stack.pop(0)
        
        order_three_cur = cur*3
        if t.does_it_close(order_three_cur,[]) == True:      
            res_close.append(words(order_three_cur))
            print(order_three_cur)     
            
            
        
            
        for move in moves:
            if len(cur+move)<=16:
                stack.append(cur+move)

    res_close = sorted(res_close)
    for i in range(len(res_close)):
        file_not_close.write(res_close[i].word+"\n")
        print(res_close[i].stdev())

    file_not_close.close()

    return res_close

def go_smaller(seq):
    '''minimalizing the knots by performing different transformation. Some transformation do not reduce
    such as AB to BA. Some transformation reduce by two such as BAB to A.
    In this method, choose a place in the sequence. Randomly apply the transformation to that part of sequence, and the start over'''
    
    #different transformations
    T_2 = ["AA","BB","CC","DD"]
    T_00 = ["AB","ACA","BCB","DC","ADA","BDB","BA","CAC","CBC","CD","DAD","DBD"]
    T_01 = ["BA","CAC","CBC","CD","DAD","DBD","AB","ACA","BCB","DC","ADA","BDB"]
    T_20 = ["A",   "B",  "D"   ,"D"   , "CA" ,  "AC",  "CB" ,  "BC",   "DA",   "AD" ,"BD",   "DB"]
    T_21 = ["BAB","ABA", "CDC","BDBDB", "ACAC", "CACA","BCBC", "CBCB", "ADAD", "DADA","BDBD", "BDBD"]
    

    for j in range(10):

        if j !=0:
            for i in range(0):
                
                qset = random.randint(0,len(T_20)-1)
                place = seq.find(T_20[qset])
                if place != -1:
                    seq_new = seq[:place]+T_21[qset]+seq[place+len(T_20[qset]):]
                    t = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))

                    clash = t.center_clash(seq_new,[])
                    if clash == False:
                        continue
                    seq = seq_new
                     
       
        l = random.randint(0,len(seq))
        seq = seq[l:]+seq[:l]
        counter = 0
        while(counter<50):

            # choose a place to apply transformations
            x = 10
            placeee = x+ random.randint(0,len(seq)-x-1)
            
           
            m=0
            # apply it 50 times
            while(m<50):



                i = random.randint(0,len(T_00)-1)
                interval = seq[placeee-x:placeee+x]
                place = placeee + interval.find(T_00[i])-x
                
            


                seq_new = seq
                if interval.find(T_00[i]) != -1:
                    #apply transformations
                    seq_new = seq_new[:place]+T_01[i]+seq_new[place+len(T_00[i]):]

                    n = 0
                    while n<4:

                        if T_2[n] in seq_new:
                            seq_new = seq_new.replace(T_2[n] ,"")
                            n=0
                            

                            continue
                        n +=1

                    t = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))
                    
                    # check if the changes made will cause crashings

                    clash = t.center_clash(seq_new,[])
                    if clash == False:
                        continue

                    seq = seq_new
                m += 1

                    
            print(counter) 
            counter += 1




    return seq

def  go_smaller_seq(seq):
    '''similar algortihm as above, slighlt different implimentation'''

    T_2 = ["AA","BB","CC","DD"]
    T_00 = ["AB","AD","BD","ACA","BCB","CDCD","BA","DA","DB","CAC","DCDC"]
    T_01 = ["BA","DA","DB","CAC","CBC","DCDC","AB","AD","BD","ACA","CDCD"]
    T_20 = ["A",   "B", "A",  "D",   "B"   ,"D"    ,"CA" ,  "AC",  "CB" ,  "BC", "CDC",  "DCD" ]
    T_21 = ["BAB","ABA","DAD","ADA","DBD","BDB", "ACAC", "CACA", "BCBC","CBCB","DCDCD","CDCDC"  ]



    for j in range(10):

        if j !=0:
            for i in range(0):
                
                qset = random.randint(0,len(T_20)-1)
                place = seq.find(T_20[qset])
                if place != -1:
                    seq_new = seq[:place]+T_21[qset]+seq[place+len(T_20[qset]):]
                    t = Tetra(np.array((0,-1,1)), np.array((0,1, 1)), np.array((-1, 0, 0))  ,np.array((-1,0,2)))   

                    clash = t.center_clash(seq_new,[])
                    if clash == False:
                        continue
                    seq = seq_new
                     
       
        l = random.randint(0,len(seq))
        seq = seq[l:]+seq[:l]
        counter = 0
        while(counter<50):

            
            x = 10
            placeee = x+ random.randint(0,len(seq)-x-1)
            
           
            m=0
            while(m<50):



                i = random.randint(0,len(T_00)-1)
                interval = seq[placeee-x:placeee+x]
                place = placeee + interval.find(T_00[i])-x
                
            
                if place <= 46 and place >= 30:
                    f=4


                seq_new = seq
                if interval.find(T_00[i]) != -1:

                    seq_new = seq_new[:place]+T_01[i]+seq_new[place+len(T_00[i]):]

                    n = 0
                    while n<4:

                        if T_2[n] in seq_new:
                            seq_new = seq_new.replace(T_2[n] ,"")
                            n=0
                            

                            continue
                        n +=1

                    t = Tetra(np.array((0,-1,1)), np.array((0,1, 1)), np.array((-1, 0, 0))  ,np.array((-1,0,2)))  
                        
                    clash = t.center_clash(seq_new,[])
                    if clash == False:
                        continue

                    seq = seq_new
                m += 1

                    
            print(counter) 
            counter += 1




    return seq

def  go_smaller(seq):
    '''similar algortihm as above, slighlt different implimentation. the transformation are applied completely randomly'''

    T_2 = ["AA","BB","CC","DD"]
    T_00 = ["AB","AD","BD","ACA","BCB","CDCD","BA","DA","DB","CAC","DCDC"]
    T_01 = ["BA","DA","DB","CAC","CBC","DCDC","AB","AD","BD","ACA","CDCD"]
    T_20 = ["A",   "B", "A",  "D",   "B"   ,"D"    ,"CA" ,  "AC",  "CB" ,  "BC", "CDC",  "DCD" ]
    T_21 = ["BAB","ABA","DAD","ADA","DBD","BDB", "ACAC", "CACA", "BCBC","CBCB","DCDCD","CDCDC"  ]
    

    for j in range(10):

        if j !=0:
            for i in range(0):
                
                qset = random.randint(0,len(T_20)-1)
                place = seq.find(T_20[qset])
                if place != -1:
                    seq_new = seq[:place]+T_21[qset]+seq[place+len(T_20[qset]):]
                    t = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))

                    clash = t.center_clash(seq_new,[])
                    if clash == False:
                        continue
                    seq = seq_new
                     

        l = random.randint(0,len(seq))
        seq = seq[l:]+seq[:l]
        counter = 0
        while(counter<250):
           

            i = random.randint(0,len(T_00)-1)
                
            seq_new = seq
            for z in range(len(T_21)):
                placee = seq.find(T_21[z])
                seq_new = seq
                if placee != -1:
                    
                    seq_new = seq_new[:placee]+T_20[z]+seq_new[placee+len(T_21[z]):]
                        
                    t = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))


                    clash = t.center_clash(seq_new,[])
                    if clash == False:
                        continue
                    seq = seq_new


           
            place = seq.find(T_00[i])


            seq_new = seq
            if seq.find(T_00[i]) != -1:

                seq_new = seq_new[:place]+T_01[i]+seq_new[place+len(T_00[i]):]

                n = 0
                while n<4:

                    if T_2[n] in seq_new:
                        seq_new = seq_new.replace(T_2[n] ,"")
                        n=0
                        

                        continue
                    n +=1

                t = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))
                    
                clash = t.center_clash(seq_new,[])
                if clash == False:
                    continue

                seq = seq_new

                    
            print(counter) 
            counter += 1




    return seq

def count(seq):
    '''count number of A,B,C and D'''
    countA = 0 
    countB = 0
    countC = 0
    countD = 0
    for s in seq:
        if s == "A":
            countA +=1
        elif s =="B":
            countB +=1
        elif s == "C":
            countC +=1
        elif s=="D":
            countD +=1
    return  countA,countB,countC,countD             
                              
def eval_diff(cur,prev):
    
    delta_x = round((cur[0] - prev[0]),3)
    delta_y = round((cur[1] - prev[1]),3)
    delta_z = round((cur[2] - prev[2]),3)
    
    return delta_x,delta_y,delta_z
     
def get_d(s):
    count = 0
    for el in s:
        if el == "D":
            count +=1
    return count

def trim_file():
    '''cut down the number of generated sequences by realizing the symmetry of A and B reflection'''
    file1 = open('orderthreeee.txt', 'r')
    file2 = open("orderthreetrimmed.txt", "w")
    line = file1.readline()
    file2.write(line)
    while line:
        line = file1.readline()
        
        if line.find("B") == -1:
            file2.write(line)
        elif line.find("A") == -1:
            continue
        elif line.find("B")>line.find("A"):
            file2.write(line)

    file1.close()
    file2.close()

def get_all_points_between(center1,center2,div):
    '''get the points between two centers'''
    v = (center2-center1)/div
    res = []
    if np.array_equal(v, np.array([0, 0, 0])):
        return [center2]
    center11 = copy.deepcopy(center1)
    for i in range(div-1):
        if i == div -2:
            pppp = "hi"
        
        center11 = center1 +(i+1) *v
        center11 = np.round(center11,4)
        res.append(center11)
    return res
def how_many_overlaps(prj_centers):
    '''count the number of overlaps if the knot is reflected on a plane'''
    all_points = []
 
    
    for i in range(1,len(prj_centers)):
        if i != len(prj_centers) -1:
            all_points.append(np.round(prj_centers[i-1],4))
        ll = get_all_points_between(prj_centers[i-1],prj_centers[i],100)
        all_points += ll

    counter = Counter(map(tuple, all_points))
    repeated_count = sum(count > 1 for count in counter.values())
    return repeated_count


def project_centers(centers,normal,point_on_plane):
    '''project the centers of the tetrahedrons onto a plane'''
    res = []
    for point in centers:
        distance = dotprod(point - point_on_plane, normal) / np.linalg.norm(normal)
        projected_point = point - distance*normal/np.linalg.norm(normal)
        res.append(projected_point)
    return res

def check_trefoil(t,seq):
    '''check if the sequence is a trefoil by counting the number of overlaps. the number of overlaps for a trefoil must always be greater than 3'''
    centers = t.reflect_seq(seq,[])[1]
    normals =        [np.array([0, 0, 1]),np.array([0, 1, 0]),np.array([1, 0, 0]),np.array([1, 1, 1]),np.array([-1, 1, 1]),np.array([-1, -1, 1]),np.array([1, -1, 1])]
    point_on_plane = np.array([0, 0, 0])
    for i in range(len(normals)):
        if i==5:
            lklskffs= 1232
        prj_centers = project_centers(centers,normals[i],point_on_plane)
        num = how_many_overlaps(prj_centers)
        if num <= 3:
            return False
    return True

def shortened_file():
    
    t1 = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))
    file2 = open('shortenedaf.txt', 'w')
    file1 = open("orderthreetrimmed.txt", "r")
    line = file1.readline()[:-1]
    
    while line:
        if check_trefoil(t1,line):
            file2.write(line+"\n")
        line = file1.readline()[:-1]

    file1.close()
    file2.close()

    
if __name__ == "__main__":
    t1 = Tetra(np.array((1, -1, -1)), np.array((1,1, -1)), np.array((0,0,-1)) ,np.array((0,0,0)))
    seq = "CBDCDCDCBADCBACBDCDCDCBADCBACBDCDCDCBADCBA"
    #print(check_trefoil(t1,seq))
    shortened_file()
    