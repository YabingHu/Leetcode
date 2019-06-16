#######################################################################
#Dynamic Programming:
#312. Burst Balloons
#Time=O(n^3),space=O(n^2)
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n=len(nums)
        nums.insert(0,1)
        nums.append(1)
        dp=[[0]*(n+2) for _ in range(n+2)]
        for l in range(1,n+1):
            for i in range(1,n-l+2):
                j=i+l-1
                for k in range(i,j+1):
                    dp[i][j]=max(dp[i][k-1]+dp[k+1][j]+nums[i-1]*nums[k]*nums[j+1],dp[i][j])
        return dp[1][n]
        


#139. Word Break
#Time=O(n^2),space=O(n)
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n= len(s)
        dp= [0] * (n+1)
        dp[0]=1
        for i in range (1,n+1):
            for j in range(i):
                if dp[j] and (s[j:i] in wordDict):
                    dp[i]=1
                    break
        return bool (dp[n])
    
#140. Word Break II
#Time=O(n^2),space=O(n)
class Solution:
    def wordBreak(self, s, wordDict):
        dp={}
        return self.dfs(s, wordDict, dp)

    def dfs(self,s,wordDict, dp):
        if s in dp: return dp[s]
        if not s: return [""]
        res=[]
        for word in wordDict:
            if s[:len(word)] !=word:continue
            memo=self.dfs(s[len(word):],wordDict,dp)
            for string in memo:
                res.append(word+("" if not string else " ") +string)
        dp[s]=res
        return res
    
#174. Dungeon Game
#Time=O(m*n),space=O(m*n)
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m=len(dungeon)
        n=len(dungeon[0])
        dp=[[float('Inf')]*(n+1) for _ in range(m+1)]
        dp[m][n-1]=1
        dp[m-1][n]=1
        for x in range(m-1,-1,-1):
            for y in range(n-1,-1,-1):
                dp[x][y]=max(1,min(dp[x+1][y],dp[x][y+1])-dungeon[x][y])
        return dp[0][0]

#115. Distinct Subsequences
#Time=O(m*n),space=O(m*n)
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        ls=len(s)
        lt=len(t)
        dp=[[0]*(ls+1) for _ in range(lt+1)]
        for i in range(ls+1):
            dp[0][i]=1
        for i in range(1,lt+1):
            for j in range(1,ls+1):
                dp[i][j]=dp[i][j-1]+(dp[i-1][j-1] if s[j-1]==t[i-1] else 0)    
        return dp[lt][ls]
        



#198. House Robber
#Time=O(n),space=O(n)
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums)==0:return 0
        dp=[0]*(len(nums))
        for i in range(len(nums)):
            a=0 if i==0 else dp[i-1]
            b=0 if i<=1 else dp[i-2]
            dp[i]=max(a,b+nums[i])
        return dp[-1]
    
#213. House Robber II
#Time=O(n),space=O(n)
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums)==0:return 0
        if len(nums)<=2:return max(nums)
        return max(self.helper(nums[1:]),self.helper(nums[:-1]))
    def helper(self,nums):
        dp=[0]*len(nums)
        for i in range(len(nums)):
            a=0 if i==0 else dp[i-1]
            b=0 if i<=1 else dp[i-2]
            dp[i]=max(a,b+nums[i])
        return dp[-1]
        
#740. Delete and Earn
#Time=O(100001+n),space=O(n+10001)
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        sums=[0]*10001
        for num in nums:
            sums[num]+=num
        dp=[0]*(len(sums))
        for i in range(len(sums)):
            a=0 if i==0 else dp[i-1]
            b=0 if i<=1 else dp[i-2]
            dp[i]=max(a,b+sums[i])
        return dp[-1]
#Time=O(n+max(nums)),space=O(max(nums))    
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        if len(nums)==0:return 0
        r=max(nums)
        m=[0]*(r+1)
        for num in nums:
            m[num]+=num
        return self.rob(m)
    def rob(self,m):
        dp1=dp2=dp=0
        for ele in m:
            dp=max(dp1,dp2+ele)
            dp2=dp1
            dp1=dp
        return dp
        
        
#790. Domino and Tromino Tiling
#Time=O(n),space=O(n)
class Solution:
    def numTilings(self, N: int) -> int:
        if N==1:return 1
        M=1e9+7
        dp=[0]*(N+1)
        dp[0]=1
        dp[1]=1
        dp[2]=2
        for i in range(3,N+1):
            dp[i]=(dp[i-1]*2+dp[i-3])%M
        return int(dp[N])

#801. Minimum Swaps To Make Sequences Increasing
#Time=O(n),space=O(n)
class Solution:
    def minSwap(self, A: List[int], B: List[int]) -> int:
        n=len(A)
        swap=[n]*n
        noSwap=[n]*n
        swap[0] = 1
        noSwap[0] = 0
        for i in range(1,n):
            if A[i]>A[i-1] and B[i]>B[i-1]:
                swap[i]=swap[i-1]+1
                noSwap[i]=noSwap[i-1]
            if A[i]>B[i-1] and B[i]>A[i-1]:
                swap[i]=min(swap[i],noSwap[i-1]+1)
                noSwap[i]=min(noSwap[i],swap[i-1])
        return min(swap[n - 1], noSwap[n - 1])
            
    
    
#70. Climbing Stairs
# Time and Space O(n) for all methods

#recursive
class Solution:
    def climbStairs(self, n: int) -> int:
        res=[0]*(n+1)
        return self.helper(n,res)

    def helper(self,n,res):
        if(n<=1): return 1
        if res[n]>0: return res[n]
        res[n]=self.helper(n-1,res)+self.helper(n-2,res)
        return res[n]
#iterative,dp     
class Solution:
    def climbStairs(self, n: int) -> int:
        dp=[0]*(n+1)
        dp[0],dp[1]=1,1
        for i in range(2,n+1):
            dp[i]=dp[i-1]+dp[i-2]
        return dp[n]
    
#DP with space=O(1)
class Solution:
    def climbStairs(self, n: int) -> int:
        dp=1
        dp1=dp2=1
        for i in range(2,n+1):
            dp=dp1+dp2
            dp2=dp1
            dp1=dp
        return dp
        
  
#746. Min Cost Climbing Stairs
# Time and Space O(n) for all methods
# Or space O(1)

#Itrative,space O(1)    
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        res=0
        dp1,dp2=0,0
        for i in range(2,len(cost)+1):
            dp=min(dp1+cost[i-1],dp2+cost[i-2])
            dp2=dp1
            dp1=dp
        return dp
    
#Itrative,space O(n)    
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        res=0
        dp=[0]*(len(cost)+1)
        dp[0],dp[1]=0,0
        for i in range(2,len(cost)+1):
            dp[i]=min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])
        return dp[len(cost)]        
       
#303. Range Sum Query - Immutable   
class NumArray:
    def __init__(self, nums: List[int]):
        self.dp = nums
        for i in range(1,len(nums)):
            self.dp[i]+=self.dp[i-1]
    def sumRange(self, i: int, j: int) -> int:
        if i==0:
            return self.dp[j]
        else:
            return self.dp[j]-self.dp[i-1]
        
#53. Maximum Subarray    
#Time O(n), space O(1)
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cursum=0
        res=float('-Inf')
        for i in range(len(nums)):
            cursum = max(cursum + nums[i], nums[i])
            res = max(res, cursum)
        return res

#121. Best Time to Buy and Sell Stock
#Time O(n),space O(1)
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy=float('Inf')
        res=0
        for i in range(len(prices)):
            buy=min(prices[i],buy)
            res=max(res,prices[i]-buy)
        return res

#787. Cheapest Flights Within K Stops
#Time=O(K*|E| |E| is ), space=O(K*n)
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        kInfCost=1e9
        dp=[[kInfCost]*(n) for _ in range(K+2)]
        dp[0][src]=0
        for i in range(1,K+2):
            dp[i][src]=0
            for p in flights:
                dp[i][p[1]]=min(dp[i][p[1]],dp[i-1][p[0]]+p[2])
        return -1 if dp[K+1][dst]>=kInfCost else dp[K+1][dst]
    
    
# 309. Best Time to Buy and Sell Stock with Cooldown   
#Time=O(n),space=O(n)->O(1)
class Solution(object):
    def maxProfit(self, prices):
        n=len(prices)
        if n<=1:return 0
        hold=-prices[0]
        rest=0
        sold=float('-Inf')
        for price in prices:
            pre_rest=rest
            rest=max(pre_rest,sold)
            sold=hold+price
            hold=max(pre_rest-price,hold)
        return max(rest,sold)    
    
class Solution(object):
    def maxProfit(self, prices):
        n=len(prices)
        if n<=1:return 0
        hold=[0]*n
        rest=[0]*n
        sold=[0]*n
        hold[0]= -prices[0]
        rest[0]=0
        sold[0]=float('-Inf')
        for i in range(1,n):
            rest[i]=max(rest[i-1],sold[i-1])
            hold[i]=max(rest[i-1]-prices[i],hold[i-1])
            sold[i]=hold[i-1]+prices[i]
        return max(sold[n-1],rest[n-1])
    
#Divide and Conquer
#169. Majority Element

#Moore Voting time:O(n) space O(1)
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        res=0
        cnt=0
        for num in nums:
            if cnt==0:
                res=num
                cnt+=1
            else:
                if num==res:
                    cnt+=1
                else:
                    cnt-=1
        return res
##############################################################################################
#LinkedList   

#19. Remove Nth Node From End of List
#Time=O(n),space=O(1)
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:return None
        pre=head
        cur=head
        for i in range(n):
            cur=cur.next
        if not cur:return head.next
        while cur.next:
            pre=pre.next
            cur=cur.next
        pre.next=pre.next.next
        return head
            







#2. Add Two Numbers
# time:O(max(m,n)), soace:O(max(m,n))+1 where m and n are length for l1,l2
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy=ListNode(-1)
        cur=dummy
        carry=0
        while l1 !=None or l2 !=None:
            d1 = l1.val if l1 else 0
            d2 = l2.val if l2 else 0
            res=d1+d2+carry
            carry = 1 if res>=10 else 0
            cur.next=ListNode(res%10)
            cur=cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        if carry==1:
            cur.next=ListNode(1)
        return dummy.next

#445. Add Two Numbers II   
time:O(max(m,n)), soace:O(max(m,n))+1+m+n where m and n are length for l1,l2
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        s1=[]
        s2=[]
        while l1!=None:
            s1.append(l1.val)
            l1=l1.next
        while l2!=None:
            s2.append(l2.val)
            l2=l2.next
        cur=ListNode((s1[-1]+s2[-1])%10)
        carry=1 if s1[-1]+s2[-1]>9 else 0
        s1.pop()
        s2.pop()
        while  s1 or  s2:
            d1=s1[-1] if s1 else 0
            d2=s2[-1] if s2 else 0
            sum=d1+d2+carry
            carry=1 if sum>9 else 0
            temp=ListNode(sum%10)
            temp.next=cur
            cur=temp
            s1.pop() if s1 else []
            s2.pop() if s2 else []
        if carry==1:
            temp=ListNode(1)
            temp.next=cur
            cur=temp
        return cur

#141. Linked List Cycle
#Time:O(n), space:O(1)
class Solution(object):
    def hasCycle(self, head):
        fast,slow=head,head
        while fast!=None and fast.next!=None:
            fast=fast.next.next
            slow=slow.next
            if fast==slow:
                return True
        return False
    
#142. Linked List Cycle II
#Time:O(n), space:O(1)
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast,slow=head,head
        while fast and fast.next:
        #while fast !=None and fast.next!=None: 
            fast=fast.next.next
            slow=slow.next
            if slow==fast:
                break
        if not fast or not fast.next: return None
        #if fast== None or fast.next == None: return None
        slow=head
        while slow!=fast:
            slow=slow.next
            fast=fast.next
        return slow

#21. Merge Two Sorted Lists
#time O(n+m),space O(1)
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy=ListNode(-1)
        cur=dummy
        while l1!=None and l2 !=None:
            if l1.val<=l2.val:
                cur.next=l1
                l1=l1.next
            else:
                cur.next=l2
                l2=l2.next
            cur=cur.next
        if l1 ==None:
            cur.next=l2
        else:
            cur.next=l1
        return dummy.next

#23. Merge k Sorted Lists
#time O(n*logn),space O(1)
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        n=len(lists)
        if n==0:return lists
        while n>1:
            k=int((n+1)/2)
            for i in range(int(n/2)):
                lists[i]=self.helper(lists[i],lists[i+k])
            n=k
        return lists[0]
    def helper(self,l1,l2):
        dummy=ListNode(-1)
        cur=dummy
        while l1 and l2:
            if l1.val<=l2.val:
                cur.next=l1
                l1=l1.next
            else:
                cur.next=l2
                l2=l2.next
            cur=cur.next
        if not l2 and l1:
            cur.next=l1
        if not l1 and l2:
            cur.next=l2
        return dummy.next

#203. Remove Linked List Elements
#Time=O(n),space=O(1)
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        if head==None:return None
        cur=head
        while cur.next:
            if cur.next.val==val:
                cur.next=cur.next.next
            else:cur=cur.next
        return head if head.val!=val else head.next

#147. Insertion Sort List
#Time=O(n^2),space=O(1)
class Solution(object):
    def insertionSortList(self, head):
        dummy=ListNode(-1)
        cur=head
        dummy.next=head
        while head and head.next:
            if head.val>head.next.val:
                cur=head.next
                pre=dummy
                while pre.next.val<cur.val:
                    pre=pre.next
                head.next=cur.next
                cur.next=pre.next
                pre.next=cur
            else:
                head=head.next
        return dummy.next

#206. Reverse Linked List
#Iterative,Time=O(n),space=O(1)
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        curr = head
        while curr:
            nextTemp = curr.next
            curr.next = prev
            prev = curr
            curr = nextTemp
        
        return prev
    
#recursive, Time=O(n),space=O(n)
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:return head
        res=self.reverseList(head.next)
        head.next.next=head
        head.next=None
        return res
        
###############################################################################################    
#Graph    
#133. Clone Graph
#DFS Solution, both time and space are O(n)
"""
# Definition for a Node.
class Node:
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors
"""
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        dict={}
        return self.helper(node,dict)
    def helper(self,node,dict):
        if node ==None: return None
        if node in dict:
            return dict[node]
        dup=Node(node.val,[])
        dict[node]=dup
        for neighbor in node.neighbors:
            clone=self.helper(neighbor,dict)
            dup.neighbors.append(clone)
        return dup
#BFS Solution, both time and space are O(n)
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return None
        dict={}
        queue=[node]
        dup=Node(node.val,[])
        dict[node]=dup
        while queue:
            t=queue.pop(0)
            for neighbor in t.neighbors:
                if neighbor not in dict:
                    dict[neighbor]=Node(neighbor.val,[])
                    queue.append(neighbor)
                dict[t].neighbors.append(dict[neighbor])
        return dup

#138. Copy List with Random Pointer
#Time and space O(n)
"""
# Definition for a Node.
class Node:
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head == None: return None
        dict={}
        node=head
        while node:
            dict[node]=Node(node.val,None,None)
            node=node.next
        node=head
        while node:
            if node.next:
                dict[node].next=dict[node.next]
            if node.random:
                dict[node].random=dict[node.random]
            node=node.next
        return dict[head]
    
#200. Number of Islands    
#Time and space O(m*n)
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m=len(grid)
        if m==0 : return 0
        n=len(grid[0])
        visited=[[0]*n for _ in range(m) ]
        res=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]=='1' and visited[i][j]==0:
                    self.helper(grid,visited,i,j)
                    res+=1
        return res
    
    def helper(self,grid,visited,x,y):
        m,n=len(grid),len(grid[0])
        if x<0 or x>=m or y<0 or y>=n or visited[x][y]==1 or grid[x][y] !='1': return
        visited[x][y]=1
        self.helper(grid,visited,x-1,y)
        self.helper(grid,visited,x+1,y)
        self.helper(grid,visited,x,y+1)
        self.helper(grid,visited,x,y-1)
 
#547. Friend Circles
#Time O(n^2),space=O(n)
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        if not M:return 0
        m=len(M)
        visited=[0]*m
        res=0
        for i in range(m):
            if visited[i]==1: continue
            self.helper(M,i,m,visited)
            res+=1
        return res
    
    def helper(self,M,i,m,visited):
        if visited[i]==1:return
        visited[i]=1
        for j in range(m):
            if M[i][j] and visited[j]==0:
                self.helper(M,j,m,visited)


#695. Max Area of Island
#Time and space O(m*n)
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m=len(grid)
        if m==0 : return 0
        n=len(grid[0])
        res=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    area=self.helper(grid,i,j,0)
                    res=max(res,area)   
        return res
    
    def helper(self,grid,x,y,area):
        m,n=len(grid),len(grid[0])
        if x<0 or x>=m or y<0 or y>=n or grid[x][y]==0: return area
        grid[x][y]=0
        area+=1
        area=self.helper(grid,x-1,y,area)
        area=self.helper(grid,x+1,y,area)
        area=self.helper(grid,x,y-1,area)
        area=self.helper(grid,x,y+1,area)
        return area

#733. Flood Fill
#Time=O(m*n),space=O(1)
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if image[sr][sc]==newColor: return image
        m,n=len(image),len(image[0])
        self.helper(image,sr,sc,image[sr][sc],newColor)
        return image
    def helper(self,image,x,y,preColor,newColor):
        m,n=len(image),len(image[0])
        if x<0 or x>=m or y<0 or y>=n: return
        if image[x][y] !=preColor: return
        image[x][y]=newColor
        self.helper(image,x+1,y,preColor,newColor)
        self.helper(image,x-1,y,preColor,newColor)
        self.helper(image,x,y+1,preColor,newColor)
        self.helper(image,x,y-1,preColor,newColor)

#841. Keys and Rooms
#Time=O(E+V),space=O(V)
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        m=len(rooms)
        if m ==1:return True
        visited=[0]*m
        self.helper(0,rooms,visited)
        if sum(visited)==m:
            return True
        else:
            return False
    def helper(self,i,rooms,visited):
        if visited[i]==1: return
        visited[i]=1
        for j in rooms[i]:
             self.helper(j,rooms,visited)
        

#802. Find Eventual Safe States
#Time=O(V+E),space=O(V+E)
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        states=['UNKNOWN']*len(graph)
        ans=[]
        for i in range(len(graph)):
            if self.helper(graph,i,states)=='SAFE':
                ans.append(i)
        return ans
    def helper(self,graph,cur,states):
        if states[cur]=='VISITING':
            states[cur]='UNSAFE'
            return states[cur]
        if states[cur] != 'UNKNOWN':
            return states[cur]
        states[cur] = 'VISITING'
        for j in graph[cur]:
            if self.helper(graph,j,states)=='UNSAFE':
                states[cur]='UNSAFE'
                return states[cur]
        states[cur]='SAFE'
        return states[cur]

#207. Course Schedule
#Time=O(n),space=O(1)
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)
        visited=[0]*numCourses
        for i in range(numCourses):
            if self.helper(i,graph,visited):
                return False
        return True
    #If there is a cycle, return TRUE
    #1 visiting/2 visited
    def helper(self,curr,graph,visited):
        if visited[curr]==1: return True
        if visited[curr]==2: return False
        visited[curr]=1
        for ele in graph[curr]:
            if self.helper(ele,graph,visited):
                return True
        visited[curr]=2
        return False
    
#210. Course Schedule II
#Time=O(n),space=O(n)
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(numCourses)]
        res=[]
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)
        visited=[0]*numCourses
        for i in range(numCourses):
            if self.helper(i,graph,visited,res):
                return []
        return res[::-1]
    #If there is a cycle, return True
    def helper(self,curr,graph,visited,res):
        if visited[curr]==1: return True
        if visited[curr]==2: return False
        visited[curr]=1
        for ele in graph[curr]:
            if self.helper(ele,graph,visited,res):
                return True
        visited[curr]=2
        res.append(curr)
        return False
    
#399. Evaluate Division
#Time=O(number of equations+e*num of queries),space=O(e)
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        res=[]
        g=collections.defaultdict(dict)
        for (x,y),v in zip(equations,values):
            g[x][y]=v
            g[y][x]=1/v
            
        for (x,y) in queries:
            res.append(self.helper(g,x,y,[]) if x in g and y in g else -1)
        return res
    def helper(self,g,x,y,visited):
        if x==y:return 1
        visited.append(x)
        for neigh in g[x]:
            if neigh in visited:continue
            visited.append(neigh)
            d=self.helper(g,neigh,y,visited)
            if d>0:return d*g[x][neigh]
        return -1
    
    
#952. Largest Component Size by Common Factor
#Time complexity: O(n*Σsqrt(A[i])) Space complexity: O(max(A))
class Solution:
    def largestComponentSize(self, A: List[int]) -> int:
        p=list(range(max(A)+1))
        for a in A:
            for i in range(2,int(math.sqrt(a)+1)):
                if a % i ==0:
                    self.union(a,i,p)
                    self.union(a,int(a/i),p)
        dict={}
        ans=1
        for a in A:
            if self.find(a,p) in dict:
                dict[self.find(a,p)]+=1
            else:dict[self.find(a,p)]=1
            ans=max(ans,dict[self.find(a,p)])
        return ans
    
    def find(self,x,p):
        while x!=p[x]:
            p[x]=p[p[x]]
            x=p[x]
        return x
    
    def union(self,x,y,p):
        p[self.find(x,p)]=p[self.find(y,p)]

#990. Satisfiability of Equality Equations
#Time=O(n),space=O(26)->O(1)
class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        UF={}
        for eq in equations:
            x,e1,e2,y=eq
            if x not in UF:UF[x]=x
            if y not in UF:UF[y]=y
            if e1=="=":
                UF[self.find(x,UF)]=UF[self.find(y,UF)]
        for eq in equations:
            x,e1,e2,y=eq
            if e1=="=" and self.find(x,UF)!=self.find(y,UF):
                return False
            if e1=="!" and self.find(x,UF)==self.find(y,UF):
                return False
        return True
    def find(self,x,UF):
        while x!=UF[x]:
            UF[x]=UF[UF[x]]
            x=UF[x]
        return x
            
##################################################################################################    
    #Binary Search
    #34. Find First and Last Position of Element in Sorted Array
    #One binary search, time O(n) in stead of O(logn) when whole array can have same number, space=O(1)
    class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        idx=self.helper(nums,0,len(nums)-1,target)
        if idx==-1: return[-1,-1]
        left,right=idx,idx
        while left>0 and nums[left-1]==nums[idx]:
            left-=1
        while right<len(nums)-1 and nums[right+1]==nums[idx]:
            right+=1
        return [left,right]
    
    def helper(self,nums,left,right,target):
        if left>right: return -1
        mid=left+int((right-left)/2)
        if target==nums[mid]: return mid
        elif nums[mid]<target:
            return self.helper(nums,mid+1,right,target)
        else: return self.helper(nums,left,mid-1,target)
        
#875. Koko Eating Bananas
# Time=O(NlogW), where N is the number of piles, and W is the maximum size of a pile
class Solution:
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        left=1
        right=1e9
        while left<=right:
            mid=left+int((right-left)/2)
            cnt=0
            for pile in piles:
                cnt+=int((pile+mid-1)/mid)
            if cnt>H:left=mid+1
            else:right=mid-1
        return left
       
#Two binary search, time=O(logn), space=O(1)
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        res=[-1]*2
        if len(nums)==0: return res
        left,right=0,len(nums)-1
        while(left<=right):
            mid=left+int((right-left)/2)
            if nums[mid]<target: left=mid+1
            else: right=mid-1
        if right+1==len(nums): return res
        if nums[right+1] !=target: return res
        res[0]=right+1
        right=len(nums)-1
        while left<=right:
            mid=left+int((right-left)/2)
            if nums[mid]<=target: left=mid+1
            else: right=mid-1
        res[1]=right
        return res

#35. Search Insert Position
#Time=O(logn), space=O(1)
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left,right=0,len(nums)-1
        while left<= right:
            mid=left+int((right-left)/2)
            if nums[mid]<target:
                left=mid+1
            else:
                right=mid-1
        return right+1
            

#704. Binary Search
#Time=O(logn), space=O(1)
class Solution:
    def search(self, nums: List[int], target: int) -> int: 
        left,right=0,len(nums)-1
        while left<=right:
            mid=left+int((right-left)/2)
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                right=mid-1
            else:
                left=mid+1
        return -1
    
#33. Search in Rotated Sorted Array
#Time O(logn), space O(n)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left=0
        right=len(nums)-1
        while left<=right:
            mid=int(left+(right-left)/2)
            if target==nums[mid]:
                return mid
            elif nums[mid]<nums[right]:
                if nums[mid]<target and nums[right]>=target:
                    left=mid+1
                else: right=mid-1
            else:
                if nums[left]<=target and nums[mid]> target:
                    right=mid-1
                else: left=mid+1
        return -1
 
#209. Minimum Size Subarray Sum
#Time=O(n), space=O(1)
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not s: return 0
        left,right,sum_=0,0,0
        res=len(nums)+1
        while right<len(nums):
            while right<len(nums) and sum_<s:
                sum_+=nums[right]
                right+=1
            while sum_>=s:
                res=min(res,right-left)
                sum_-=nums[left]
                left+=1
        res=0 if res==len(nums)+1 else res
        return res

#Binary search method
#Time=O(nlgn), space=O(n)
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        res=float('Inf')
        n=len(nums)
        sums=[0]*(n+1)
        for i in range(1,n+1):
            sums[i]=sums[i-1]+nums[i-1]
        for i in range(n):
            left=i+1
            right=n
            t=sums[i]+s
            while left<=right:
                mid=left+int((right-left)/2)
                if sums[mid]<t: left=mid+1
                else:right=mid-1
            if left==n+1: break
            res=min(res,left-i)
        res=0 if res==float('Inf') else res
        return res

#852. Peak Index in a Mountain Array    
#Time=O(n).space=O(1)
class Solution:
    def peakIndexInMountainArray(self, A: List[int]) -> int:
        for i in range(1,len(A)+1):
            if A[i-1]>A[i]:
                break
        return i-1
    
#Binary search method
#Time=O(nlgn), space=O(1)
class Solution:
    def peakIndexInMountainArray(self, A: List[int]) -> int:
        lo, hi = 0, len(A) - 1
        while lo <= hi:
            mi = lo+int((hi-lo)/2)
            if A[mi] < A[mi + 1]:
                lo = mi + 1
            else:
                hi = mi-1
        return lo

#29. Divide Two Integers
#Time and space=(nlogn)
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        if (dividend>0 and divisor>0) or  (dividend<0 and divisor<0):
            sign=1
        else:
            sign=-1
        dividend=abs(dividend)
        divisor=abs(divisor)
        if dividend<divisor or dividend==0:return 0
        res=self.helper(dividend,divisor)
        res=res if sign==1 else -res
        return min(max(-2147483648, res), 2147483647)
    
    def helper(self,dividend,divisor):
        if dividend<divisor:return 0
        sum_=divisor
        multiple=1
        while sum_+sum_<=dividend:
            sum_+=sum_
            multiple+=multiple
        return multiple+self.divide(dividend-sum_,divisor)

#349. Intersection of Two Arrays
#Time=O(n), space=O(n)
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dict={}
        res=[]
        for num in nums1:
            if num in dict:
                dict[num]+=1
            else:
                dict[num]=1
        for num in nums2:
            if num in dict:
                res.append(num)
        res=list(set(res))
        return res
            
#349. Intersection of Two Arrays
#Time=O(nlogn), space=O(n)
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums2.sort()
        res=[]
        for num in nums1:
            if self.helper(num,nums2):
                res.append(num)
        return list(set(res))
    
    def helper(self,target,nums2):
        left,right=0,len(nums2)-1
        while left<=right:
            mid=left+int((right-left)/2)
            if target==nums2[mid]:
                return True
            elif target>nums2[mid]:
                left=mid+1
            else:
                right=mid-1
        return False
    
#278. First Bad Version        
#Time=O(logn),space=O(1)
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left=1
        right=n
        while left<=right:
            mid=left+int((right-left)/2)
            if isBadVersion(mid):
                right=mid-1
            else:
                left=mid+1
        return left

#658. Find K Closest Elements
#Time=O(logn),space=O(1)
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left=0
        right=len(arr)-k
        while left< right:
            mid=left+int((right-left)/2)
            if x-arr[mid]>arr[mid+k]-x:
                left=mid+1
            else: right=mid
        return arr[left:left+k]
    
#240. Search a 2D Matrix II
#Time=O(log(n!)), space=O(1)
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        m=len(matrix)
        if m==0: return False
        n=len(matrix[0])
        if n==0:
            return False
        if matrix[0][0]>target or matrix[-1][-1]<target: return False
        for i in range(m):
            if self.helper(matrix[i],target): return True
        return False
        
    def helper(self,nums,target):
        left=0
        right=len(nums)-1
        while left<=right:
            mid=left+int((right-left)/2)
            if target==nums[mid]:
                return True
            elif nums[mid]<target:
                left=mid+1
            else: right=mid-1
        return False

#410. Split Array Largest Sum
#Time=O(log(sum(nums))*n),space=O(1)
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        l=max(nums)
        r=sum(nums)+1
        while l<r:
            limit=l+int((r-l)/2)
            if self.helper(nums,limit)>m:
                l=limit+1
            else: r=limit
        return l
    def helper(self,nums,limit):
        sum_=0
        groups=1
        for num in nums:
            if sum_+num>limit:
                sum_=num
                groups+=1
            else: sum_+=num
        return groups

#Dynamical programming method   
#Time=O(m*n^2), space=O(mn)
#Time Limit Exceeded
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        n=len(nums)
        sums=[0]*n
        dp=[[float('Inf')]*n for _ in range(m+1)]
        sums[0]=nums[0]
        for i in range(1,n):
            sums[i]=nums[i]+sums[i-1]
        for i in range(n):
            dp[1][i]=sums[i]
        for i in range(2,m+1):
            for j in range(i-1,n):
                for k in range(j):
                    dp[i][j]=min(dp[i][j],max(dp[i-1][k],sums[j]-sums[k]))
        return dp[m][n-1]
    
#4. Median of Two Sorted Arrays
#O(log(min(n1,n2))),space=O(1)
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1=len(nums1)
        n2=len(nums2)
        if n1>n2:
            return self. findMedianSortedArrays(nums2,nums1)
        k=int((n1+n2+1)/2)
        l=0
        r=n1
        while l<r:
            m1=l+int((r-l)/2)
            m2=k-m1
            if nums1[m1]<nums2[m2-1]:
                l=m1+1
            else:
                r=m1
        m1=l
        m2=k-l
        c1=max(-float('Inf') if m1<=0 else nums1[m1-1],-float('Inf') if m2<=0 else nums2[m2-1])
        if (n1+n2)%2==1:
            return float(c1)
        c2=min(float('Inf') if m1>=n1 else nums1[m1],float('Inf') if m2>=n2 else nums2[m2])
        return (c1+c2)/2.0

    
#270. Closest Binary Search Tree Value
#Time=O(n),space=O(1)
class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        res=root.val
        while root!=None:
            if abs(res-target)>=abs(root.val-target):
                res=root.val
            if root.val>target:
                root=root.left
            else: root=root.right
        return res

#965. Univalued Binary Tree
#Time=O(n), space=O(n)
class Solution:
    def isUnivalTree(self, root: TreeNode) -> bool:
        val=root.val
        return self.helper(root,val)
    def helper(self,root,val):
        if not root :return True
        if root.val != val:return False
        return self.helper(root.left,val) and self.helper(root.right,val)
##############################################################################################333
#Binary serach Tree
#98. Validate Binary Search Tree
#Time =O(n), space=O(n)
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if root==None:return True
        return self.helper(root,float('-Inf'),float('Inf'))
    def helper(self,root,left,right):
        if root==None:return True
        if root.val>=right or root.val<=left:
            return False
        return self.helper(root.left,left,root.val) and self.helper(root.right,root.val,right)
    
#530. Minimum Absolute Difference in BST
#Time =O(n), space=O(n)
class Solution(object):
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        res=float('Inf')
        return self.helper(root,float('-Inf'),float('Inf'),res)
    def helper(self,root,low,high,res):
        if not root:return res
        temp=min(root.val-low, high-root.val)
        res=min(res,temp)
        return min(self.helper(root.left,low,root.val,res),self.helper(root.right,root.val,high,res))
 
#700. Search in a Binary Search Tree
#Time =O(n), space=O(n)
class Solution(object):
    def searchBST(self, root, val):
        if not root:return None
        if root.val>val:
            return self.searchBST(root.left,val)
        elif root.val<val:
            return self.searchBST(root.right,val)
        else:return root

#701. Insert into a Binary Search Tree
#Time=O(H),Space=O(H)
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if root==None:return TreeNode(val)
        if root.val>val:root.left=self.insertIntoBST(root.left,val)
        else: root.right=self.insertIntoBST(root.right,val)
        return root
   
#230. Kth Smallest Element in a BST
#Time=O(H+k), time=O(H+k)
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        if root==None:return None
        s=[]
        res=[]
        cnt=0
        cur=root
        while s or cur:
            while cur:
                s.append(cur)
                cur=cur.left
            cur=s.pop()
            res.append(cur.val)
            cnt+=1
            if cnt==k:
                return res[-1]
            cur=cur.right
            
            
#108. Convert Sorted Array to Binary Search Tree
#Time=O(n),space=O(n)
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return self.helper(nums,0,len(nums)-1)
    def helper(self,nums,left,right):
        if left > right:
            return
        mid=left+(right-left)//2
        root=TreeNode(nums[mid])
        root.left=self.helper(nums,left,mid-1)
        root.right=self.helper(nums,mid+1,right)
        return root
    
#230. Kth Smallest Element in a BST
#Time=O(n),space=O(n)
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        dict={}
        if root==None:return []
        s=[]
        res=[]
        cur=root
        mx=0
        while s or cur:
            while cur:
                s.append(cur)
                cur=cur.left
            cur=s.pop()
            if cur.val not in dict:
                dict[cur.val]=1
            else:dict[cur.val]+=1
            mx=max(mx,dict[cur.val])
            cur=cur.right
        for ele in dict:
            if mx==dict[ele]:
                res.append(ele)
        return res
  

#450. Delete Node in a BST
#Time=O(logn), space=O(n)
class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if root==None:return None
        if root.val>key:
            root.left=self.deleteNode(root.left,key)
        elif root.val<key:
            root.right=self.deleteNode(root.right,key)
        else:
            if (not root.left) or (not root.right):
                root=root.right if not root.left else root.left
            else:
                cur=root.right
                while cur.left:cur=cur.left
                root.val=cur.val
                root.right=self.deleteNode(root.right,cur.val)
        return root

#99. Recover Binary Search Tree
#Time=O(n),space=O(n)
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        pre,first,second=None,None,None
        s=[]
        cur=root
        while s or cur:
            while cur:
                s.append(cur)
                cur=cur.left
            cur=s.pop()
            if pre:
                if pre.val>cur.val:
                    if not first:
                        first=pre
                    second=cur
            pre=cur
            cur=cur.right
        first.val, second.val =second.val,first.val
#########################################################################
#Tree:
#94. Binary Tree Inorder Traversal
#Time=O(n), space=O(n) for all methods.
#Recursive solution
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res=[]
        if(not root): return res
        self.helper(root,res)
        return res
    def helper(self,root,res):
        if(not root): return
        self.helper(root.left,res)
        res.append(root.val)
        self.helper(root.right,res)
        
#Using stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res=[]
        if(root==None): return res
        s=[]
        cur=root
        while cur or s:
            while cur:
                s.append(cur)
                cur=cur.left
            cur=s.pop()
            res.append(cur.val)
            cur=cur.right
        return res
#144. Binary Tree Preorder Traversal
#Time O(n), space=O(n) for both methods
#Iterative:    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        s=[root]
        res=[]
        if root==None:
            return res
        while s :
            cur=s.pop()
            res.append(cur.val)
            if cur.right != None: s.append(cur.right)
            if cur.left != None: s.append(cur.left)
        return res
#Recursive:
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:     
        res=[]
        if root==None: return res
        self.helper(root,res)
        return res
    def helper(self,root,res):
        if root==None: return
        res.append(root.val)
        self.helper(root.left,res)
        self.helper(root.right,res)
            

#589. N-ary Tree Preorder Traversal
#Time O(n), space=O(n) for both methods
#Iterative
class Solution:
def preorder(self, root: 'Node') -> List[int]:
    res=[]
    if root==None: return res
    s=[root]
    while s:
        cur=s.pop()
        res.append(cur.val)
        for i in range(len(cur.children)-1,-1,-1):
            s.append(cur.children[i])
    return res
#Recursive
class Solution:
def preorder(self, root: 'Node') -> List[int]:
    res=[]
    if root==None: return res
    self.helper(root,res)
    return res
def helper(self,root,res):
    if root==None: return 
    res.append(root.val)
    for child in root.children:
        self.helper(child,res)

#145. Binary Tree Postorder Traversal
#Time=O(n), space=O(n)
#Iterative
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res=[]
        if root==None: return res
        s=[root]
        while s:
            cur=s.pop()
            res.append(cur.val)
            if cur.left:
                s.append(cur.left)
            if cur.right:
                s.append(cur.right)
        return res[::-1]
#Recursive   
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res=[]
        if root==None: return res
        self.helper(root,res)
        return res
    def helper(self,root,res):
        if root==None: return 
        self.helper(root.left,res)
        self.helper(root.right,res)
        res.append(root.val)
        
#590. N-ary Tree Postorder Traversal
#Time=O(n), space=O(n)
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        res=[]
        if root==None:return res
        s=[root]
        while s:
            cur=s.pop()
            res.append(cur.val)
            for child in cur.children:
                s.append(child)
        return res[::-1]
                

#114. Flatten Binary Tree to Linked List
#Time=O(n), spcae=O(1)
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.pre = None
        self.helper(root)

    def helper(self,root):
        if not root:return None
        self.helper(root.right)
        self.helper(root.left)
        root.right = self.pre
        root.left = None
        self.pre = root

#102. Binary Tree Level Order Traversal        
#Time=O(n), space=O(n) for both BFS and DFS solutions
#BFS solution
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        if root == None:
            return res
        q=[root]
        while q:
            res_sub=[]
            for i in range(len(q)):
                t=q.pop(0)
                res_sub.append(t.val)
                if t.left !=None: 
                    q.append(t.left)
                if t.right !=None:
                    q.append(t.right)
            res.append(res_sub)
        return res

#DFS solution
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        if root==None:return res
        self.helper(0,root,res)
        return res
    def helper(self,depth,root,res):
        if root==None:return 
        if len(res)<=depth:
            res.append([])
        res[depth].append(root.val)
        self.helper(depth+1,root.left,res)
        self.helper(depth+1,root.right,res)
    
    
    
#99. Recover Binary Search Tree
#Time=O(n),space=O(1)
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        pre,first,second=None,None,None
        s=[]
        cur=root
        while s or cur:
            while cur:
                s.append(cur)
                cur=cur.left
            cur=s.pop()
            if pre:
                if pre.val>cur.val:
                    if not first:
                        first=pre
                    second=cur
            pre=cur
            cur=cur.right
        first.val,second.val=second.val,first.val

#501. Find Mode in Binary Search Tree
#Time=O(n),space=O(n)
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        if root==None:return []
        s=[]
        cur=root
        m={}
        mx=0
        res=[]
        while s or cur:
            while cur:
                s.append(cur)
                cur=cur.left
            cur=s.pop()
            if cur.val in m:
                m[cur.val]+=1
            else:m[cur.val]=1
            mx=max(mx,m[cur.val])
            cur=cur.right
        
        for item in m:
            if m[item]==mx:
                res.append(item)
        return res

#DFS solution:
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        self.helper(root,0,res)
        return res
    def helper(self,root,depth,res):
        if root==None: return
        while len(res)<=depth:
            res.append([])
        res[depth].append(root.val)
        self.helper(root.left,depth+1,res)
        self.helper(root.right,depth+1,res)
        
#107. Binary Tree Level Order Traversal II
#Time=O(n), space=O(n) for both BFS and DFS solutions
#BFS solution
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res=[]
        if root==None:return res
        s=[root]
        while s:
            res_sub=[]
            for i in range(len(s)):
                cur=s.pop(0)
                res_sub.append(cur.val)
                if cur.left:
                    s.append(cur.left)
                if cur.right:
                    s.append(cur.right)
            res.append(res_sub)
        return res

#DFS, preorder
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        res=[]
        self.helper(root,0,res)
        return res[::-1]
    def helper(self,root,depth,res):
        if root==None: return
        while len(res)<=depth:
            res.append([])
        res[depth].append(root.val)
        self.helper(root.left,depth+1,res)
        self.helper(root.right,depth+1,res)
 
#108. Convert Sorted Array to Binary Search Tree
#Time=O(logn),space=O(n)
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        return self.helper(nums,0,len(nums)-1)
    def helper(self,nums,left,right):
        if left>right: return None
        mid=left+int((right-left)/2)
        cur=TreeNode(nums[mid])
        cur.left=self.helper(nums,left,mid-1)
        cur.right=self.helper(nums,mid+1,right)
        return cur

#129. Sum Root to Leaf Numbers
#Time=O(n),space=O(1)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        return self.helper(root,0)
    def helper(self,root,prefix):
        if root==None: return 0
        if root.left ==None and root.right==None:
            return prefix*10+root.val  
        return self.helper(root.left,prefix*10+root.val)+self.helper(root.right,prefix*10+root.val)

#669. Trim a Binary Search Tree
#Time=O(n),space=0(1)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
        if root==None: return None
        if L<=root.val and R>=root.val:
            root.left=self.trimBST(root.left,L,R)
            root.right=self.trimBST(root.right,L,R)
        elif root.val<L:
            root=self.trimBST(root.right,L,R)
        else:
            root=self.trimBST(root.left,L,R)
        return root
    
#100. Same Tree
#Time=O(n),space=O(n)
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q: return True
        if not p or not q or p.val!=q.val:return False
        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
        
#572. Subtree of Another Tree
#Time=O(n),space=O(n)
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if s==None:return False
        if self.helper(s,t):
            return True
        return self.isSubtree(s.left,t) or self.isSubtree(s.right,t)
    def helper(self,s,t):
        if not s and not t:return True
        if (not s and t) or (s and not t) or s.val!=t.val:
            return False
        return self.helper(s.left,t.left) and self.helper(s.right,t.right)
    
#965. Univalued Binary Tree    
#Time=O(n),space=O(1)
class Solution:
    def isUnivalTree(self, root: TreeNode) -> bool:
        if root==None:return True
        target=root.val
        if self.helper(root,target):
            return True
        return False
    def helper(self,root,target):
        if root==None:return True
        if root.val!=target:
            return False
        return self.helper(root.left,target) and self.helper(root.right,target)

#872. Leaf-Similar Trees
#Time and space =O(T1+T2), where T1 and T2 is the length of the trees
class Solution:
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        return self.helper(root1)==self.helper(root2)
    def helper(self,root):
        if root==None:return []
        if not root.left and not root.right:
            return [root.val]
        return self.helper(root.left)+self.helper(root.right)
    
#987. Vertical Order Traversal of a Binary Tree
#Time=O(n*logn),Space=O(n)
class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        seen = collections.defaultdict(lambda: collections.defaultdict(list))
        def dfs(node, x, y):
            if node:
                seen[x][y].append(node)
                dfs(node.left, x-1, y-1)
                dfs(node.right, x+1, y-1)
        dfs(root,0,0)
        ans = []
        for x in sorted(seen):
            report = []
            for y in sorted(seen[x],reverse=True):
                report.extend(sorted(node.val for node in seen[x][y]))
            ans.append(report)
        return ans

#814. Binary Tree Pruning
#Time=O(n),Space=O(H),H is the height of the tree
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        if root==None:return None
        root.left=self.pruneTree(root.left)
        root.right=self.pruneTree(root.right)
        if root.left==None and root.right==None and root.val==0:
            return None
        else:return root
    
#437. Path Sum III
#Time=O(n),Space=O(H),H is the height of the tree
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        if root==None:return 0
        return self.helper(root,0,sum)+self.pathSum(root.left,sum)+self.pathSum(root.right,sum)
    def helper(self,root,pre,sum):
        if root==None:return 0
        cur=pre+root.val
        return (cur==sum) +self.helper(root.left,cur,sum)+self.helper(root.right,cur,sum)

#124. Binary Tree Maximum Path Sum
#Time=O(n),Space=O(H),H is the height of the tree
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        if root==None:return 0
        self.ans=float('-Inf')
        self.helper(root)
        return self.ans
    def helper(self,root):
        if root==None:return float('-Inf')
        l=max(0,self.helper(root.left))
        r=max(0,self.helper(root.right))
        self.ans = max(self.ans, root.val + l + r)
        return root.val + max(l, r)
    
#543. Diameter of Binary Tree
#Time=O(n),Space=O(H),H is the height of the tree
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans=0
        self.helper(root)
        return self.ans
    def helper(self,root):
        if root==None:return -1
        l=self.helper(root.left)+1
        r=self.helper(root.right)+1
        self.ans=max(self.ans,l+r)
        return max(l,r)

#687. Longest Univalue Path   
#Time=O(n), space=O(logn)->O(n) worst case
class Solution:
    def longestUnivaluePath(self, root: TreeNode) -> int:
        self.res=0
        self.helper(root)
        return self.res
    def helper(self,root):
        if root==None:return 0
        left=self.helper(root.left)
        right=self.helper(root.right)
        left=left+1 if (root.left and root.left.val==root.val) else 0
        right=right+1 if (root.right and root.right.val==root.val) else 0
        self.res=max(self.res,left+right)
        return max(left,right) 
    
#257. Binary Tree Paths
#Time=O(n), space=O(logn)->O(n) worst case
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        res=[]
        self.helper(root,res,'')

        return res
    def helper(self,root,res,out):
        if not root:return 
        if not root.left and not root.right:
            res.append(out+str(root.val))
        if root.left:
            self.helper(root.left,res,out+str(root.val)+'->')
        if root.right:
            self.helper(root.right,res,out+str(root.val)+'->')

#508. Most Frequent Subtree Sum            
#Time=O(n),space=O(n)
class Solution:
    def findFrequentTreeSum(self, root: TreeNode) -> List[int]:
        res=[]
        m={}
        self.cnt=0
        self.helper(root,m,self.cnt,res)
        return res
    def helper(self,root,m,cnt,res):
        if root==None:return 0
        left=self.helper(root.left,m,self.cnt,res)
        right=self.helper(root.right,m,self.cnt,res)
        sum=left+right+root.val
        if sum in m:
            m[sum]+=1
        else:m[sum]=1
        if m[sum]>=self.cnt:
            if m[sum]>self.cnt:
                res.clear()
            res.append(sum)
            self.cnt=m[sum]
        return sum            
            
#297. Serialize and Deserialize Binary Tree
#Time=O(n),space=O(n)
class Codec:
    def serialize(self, root):
        self.out=''
        self.helper(root,self.out)
        return self.out
    def helper(self,root,out):
        if not root:self.out+='None,'
        else:
            self.out+=str(root.val)+','
            self.helper(root.left,self.out)
            self.helper(root.right,self.out)

    def deserialize(self, data):
        data_list=data.split(',')
        root=self.helper2(data_list)
        return root
    def helper2(self,data_list):
        if data_list[0]=='None':
            data_list.pop(0)
            return
        root=TreeNode( data_list.pop(0))
        root.left=self.helper2(data_list)
        root.right=self.helper2(data_list)
        return root
    
#572. Subtree of Another Tree    
#Time=O(m*n), space=O(n), n=# nodes in s
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if not s and not t :return True
        if not s or not t : return False
        if self.helper(s,t):return True
        else:
            return self.isSubtree(s.left,t) or self.isSubtree(s.right,t) 
    def helper(self,s,t):
        if not s and not t :return True
        if not s or not t or s.val != t.val:return False
        return self.helper(s.left,t.left) and self.helper(s.right,t.right)
            
###############################################################################            
#Search:
#46. Permutations
#Time=O(n!), space=O(n!), in fact recusion has the form T(n)=T(n-1）+T（n-2）+.... has the time complexity O(2^(n)),here since 
#this line: if visited[i]==1: continue , it reduce the time complexity from O(2^n) to O(n!)
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res=[]
        out=[]
        visited=[0]*len(nums)
        self.helper(nums,0,visited,out,res)
        return res
    
    def helper(self,nums,level,visited,out,res):
        if level==len(nums):
            res.append(out[:])
            return
        for i in range(len(nums)):
            if visited[i]==1: continue
            visited[i]=1
            out.append(nums[i])
            self.helper(nums,level+1,visited,out,res)
            out.pop()
            visited[i]=0
            
#Time=O(n!), space=O(n)
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res=[]
        self.helper(nums,0,res)
        return res
    def helper(self,nums,start,res):
        if start==len(nums):
            res.append(nums[:])
            return
        for i in range(start,len(nums)):
            nums[start],nums[i]=nums[i],nums[start]
            self.helper(nums,start+1,res)
            nums[start],nums[i]=nums[i],nums[start]

#47. Permutations II          
#Time=O(n!), space=O(n!)
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res=[]
        out=[]
        visited=[0]*len(nums)
        nums.sort()
        self.helper(nums,0,visited,out,res)
        return res
    def helper(self,nums,level,visited,out,res):
        if level==len(nums):
            res.append(out[:])
            return
        for i in range(len(nums)):
            if visited[i]==1: continue
            if i>0 and nums[i]==nums[i-1] and visited[i-1]==0: continue
            visited[i]=1
            out.append(nums[i])
            self.helper(nums,level+1,visited,out,res)
            out.pop()
            visited[i]=0

#784. Letter Case Permutation
#Time=O(n* 2^l, l the number of letters in a string), space=O(n)+O(n*2^l)
class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:
        ans = []
        self.helper(list(S), 0, len(S),ans)
        return ans
    def helper(self,S, i, n,ans):
        if i == n:
            ans.append(''.join(S))
            return
        self.helper(S, i + 1, n,ans)      
        if not S[i].isalpha(): return      
        S[i] = chr(ord(S[i]) ^ (1<<5))
        self.helper(S, i + 1, n,ans)
        S[i] = chr(ord(S[i]) ^ (1<<5))

#22. Generate Parentheses
#Time=O(n!)， seems like O(2^n) but using Catalan number theory, get O(n!), space=O(n)
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res=[]
        self.helper(n,n,'',res)
        return res
    def helper(self,left,right,out,res):
        if left<0 or right<0 or left>right: return
        if left==0 and right==0:
            res.append(out)
            return
        self.helper(left-1,right,out+'(',res)
        self.helper(left,right-1,out+')',res)

#72. Edit Distance
#Time=O(m*n),space=O(m*n)
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1=len(word1)+1
        n2=len(word2)+1
        dp=[[-1]*n2 for i in range(n1)]
        for i in range(n1):
            dp[i][0]=i
        for i in range(n2):
            dp[0][i]=i
        for i in range(1,n1):
            for j in range(1,n2):
                if word1[i-1]==word2[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])+1
        return dp[n1-1][n2-1]

    
#207. Course Schedule
#Finding cycles O(n^2) -> Topological sort O(n), space=O(n)
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)
        visited=[0]*numCourses
        for i in range(numCourses):
            if self.helper(i,graph,visited):
                return False
        return True
    #If there is a cycle, return False
    def helper(self,curr,graph,visited):
        if visited[curr]==1: return True
        if visited[curr]==2: return False
        visited[curr]=1
        for ele in graph[curr]:
            if self.helper(ele,graph,visited):
                return True
        visited[curr]=2
        return False
    
#210. Course Schedule II
#Time=O(E+V)->O(n), space=O(n)
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(numCourses)]
        res=[]
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)
        visited=[0]*numCourses
        for i in range(numCourses):
            if self.helper(i,graph,visited,res):
                return []
        return res[::-1]
    #If there is a cycle, return False
    def helper(self,curr,graph,visited,res):
        if visited[curr]==1: return True
        if visited[curr]==2: return False
        visited[curr]=1
        for ele in graph[curr]:
            if self.helper(ele,graph,visited,res):
                return True
        visited[curr]=2
        res.append(curr)
        print(curr)
        return False
    
#17. Letter Combinations of a Phone Number
#Time=O(4^n),space=O(4^n+n)
class Solution:
    def letterCombinations(self, digits):
        if not digits:return []
        dict = {"1":"", "2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs","8":"tuv","9":"wxyz","10":" "}
        res = []        
        self.dfs(dict,digits,0,"",res)
        return res
            
    def dfs(self,dict,string,level,out,res):
        if level ==len(string):
            res.append(out)
            return
        for i in dict[string[level]]:
            self.dfs(dict,string,level+1,out+i,res)
    
#39. Combination Sum
#Time=(2^n),space=O(k*n) where k is the number of answers
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res=[]
        self.helper(candidates,target,[],0,res)
        return res
    def helper(self,candidates,target,out,index,res):
        if target<0:return
        if target==0:
            res.append(out)
        for i in range(index,len(candidates)):
            self.helper(candidates,target-candidates[i],out+[candidates[i]],i,res)
        
    
#40. Combination Sum II
#Time=(2^n),space=O(k*n) where k is the number of answers
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res=[]
        candidates.sort()
        self.helper(candidates,target,[],0,res)
        return res
    def helper(self,candidates,target,out,index,res):
        if target<0:return
        if target==0:
            res.append(out)
        for i in range(index,len(candidates)):
            if i>index and candidates[i]==candidates[i-1]:continue
            self.helper(candidates,target-candidates[i],out+[candidates[i]],i+1,res)
            
#77. Combinations
#Time=O(k*C^n_{k}), space=O(C^n_{k})
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res=[]
        self.helper(n,k,1,[],res)
        return res
    def helper(self,n,k,index,out,res):
        if len(out)==k:
            res.append(out)
            return
        for i in range(index,n+1):
            self.helper(n,k,i+1,out+[i],res)
        return
            
#78. Subsets
#Time=O(n!),space=(n!)
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res=[]
        self.helper(nums,0,[],res)
        return res
    def helper(self,nums,start,out,res):
        res.append(out)
        for i in range(start,len(nums)):
            self.helper(nums,i+1,out+[nums[i]],res)
        return
            
#90. Subsets II
#Time=O(n!),space=(n!)
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res=[]
        nums.sort()
        self.helper(nums,[],0,res)
        return res
    def helper(self,nums,out,index,res):
        res.append(out)
        for i in range(index,len(nums)):
            if i>index and nums[i]==nums[i-1]:continue
            self.helper(nums,out+[nums[i]],1+i,res)
                     
#79. Word Search
#Time=O(m*n*4^l, where l in the length of a word), space=(m*n+l)
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m=len(board)
        if m==0:return False
        n=len(board[0])
        if n==0:return False
        visited=[[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if self.helper(board,visited,word,0,i,j):
                    return True
        return False
 
    def helper(self,board,visited,word,level,x,y):
        if level==len(word):return True
        m=len(board)
        n=len(board[0])
        if x>=m or x<0 or y>=n or y<0 or visited[x][y]==1 or word[level]!=board[x][y]:
            return False
        visited[x][y]=1
        res= self.helper(board,visited,word,level+1,x+1,y) or self.helper(board,visited,word,level+1,x-1,y) or self.helper(board,visited,word,level+1,x,y-1) or self.helper(board,visited,word,level+1,x,y+1)
        visited[x][y]=0
        return res
        
#22. Generate Parentheses
#Time=O(2^n), space=O(n)
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res=[]
        self.helper(n,n,res,'')
        return res
    def helper(self,left,right,res,out):
        if left<0 or right <0 or left>right:return
        if left==0 and right==0:
            res.append(out)
        self.helper(left-1,right,res,out+"(")
        self.helper(left,right-1,res,out+")")
                
#51. N-Queens
#Time=O(n!),space=O(n)
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res=[]
        queenCol=[-1]*n
        self.helper(0,queenCol,res)
        return res
    def helper(self,curRow,queenCol,res):
        n=len(queenCol)
        if curRow==n:
            tmp='.'*n
            out=[]
            for i in range(n):
                out.extend([tmp[:queenCol[i]]+"Q"+tmp[queenCol[i]+1:]])
            res.append(out)
            return
        for i in range(n):
            if self.isValid(queenCol,curRow,i):
                queenCol[curRow]=i
                self.helper(curRow+1,queenCol,res)
                queenCol[curRow]=-1
    def isValid(self,queenCol,row,col):
        for i in range(row):
            if col==queenCol[i] or abs(row-i)==abs(col-queenCol[i]):
                return False
        return True

#52. N-Queens II
#Time=O(n!),space=O(n)      
class Solution:
    def totalNQueens(self, n: int) -> int:
        self.cnt=0
        queenCol=[-1]*n
        self.helper(0,queenCol,self.cnt)
        return self.cnt
    def helper(self,curRow,queenCol,cnt):
        n=len(queenCol)
        if curRow==n:
            self.cnt+=1
            return
        for i in range(n):
            if self.isValid(queenCol,curRow,i):
                queenCol[curRow]=i
                self.helper(curRow+1,queenCol,self.cnt)
                queenCol[curRow]=-1
    def isValid(self,queenCol,row,col):
        for i in range(row):
            if col==queenCol[i] or abs(row-i)==abs(col-queenCol[i]):
                return False
        return True
                
#Array
#54. Spiral Matrix
#Time=O(n), space=O(n)
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix: return []
        R, C = len(matrix), len(matrix[0])
        seen = [[False] * C for _ in range(R)]
        ans = []
        dr = [0, 1, 0, -1]
        dc = [1, 0, -1, 0]
        r = c = di = 0
        for i in range(R * C):
            ans.append(matrix[r][c])
            seen[r][c] = True
            cr, cc = r + dr[di], c + dc[di]
            if 0 <= cr < R and 0 <= cc < C and seen[cr][cc]==0:
                r, c = cr, cc
            else:
                di = (di + 1) % 4
                r, c = r + dr[di], c + dc[di]
        return ans

#55. Jump Game
#Dynamical programming, time=O(n),space=O(n)
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        dp=[0]*len(nums)
        dp[0]=0
        for i in range(1,len(nums)):
            dp[i]=max(dp[i-1],nums[i-1])-1
            if dp[i]<0:
                return False
        return True

#Greedy
# time=O(n),space=O(1)
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums)<2: return True
        reach=0
        for i in range(len(nums)):
            if i > reach or reach >= len(nums) - 1:
                break
            reach=max(nums[i]+i,reach)
        return reach >= len(nums) - 1
    
#228. Summary Ranges
#Time=O(n), space=O(n)
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        i=0
        n=len(nums)
        res=[]
        while i<n:
            j=1
            while i+j<n and nums[i+j]-nums[i]==j:
                j+=1
            if j<=1:
                 res.append(str(nums[i]))
            else:
                 res.append(str(nums[i])+"->"+ str(nums[i+j-1]))
            i+=j
        return res
    
#189. Rotate Array
#Time=O(n), space=O(n)
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        temp=nums[:]
        for i in range(len(nums)):
            nums[(i+k)%len(nums)]=temp[i]

#Time=O(n),space=O(1)
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n=len(nums)
        if n==0 or k%len(nums)==0:return
        nums[0:(n-k)]=nums[0:(n-k)][::-1]
        nums[(n-k):]=nums[(n-k):][::-1]
        nums[:]=nums[::-1]
  
#628. Maximum Product of Three Numbers
#Time=O(nlogn) for sorting, space=O(logn) for sorting
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        n=len(nums)
        if n==3: return nums[0]*nums[1]*nums[2]
        nums.sort()
        p= nums[0] * nums[1] * nums[-1]
        return max(p, nums[n - 1] * nums[n - 2] * nums[n - 3])

#1. Two Sum
#Time=O(n),space=O(n)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict={}
        res=[]
        for i in range(len(nums)):
            temp=target-nums[i]
            if temp in dict:
                res.append(i)
                res.append(dict[temp])
                break
            dict[nums[i]]=i
        return res
    
#15. 3Sum
#
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res=[]
        nums.sort()
        if len(nums)==0 or nums[0]>0 or nums[-1]<0: return res
        for i in range(len(nums)):
            if nums[i]>0:break
            if i>0 and nums[i]==nums[i-1]:continue
            target=0-nums[i]
            left=i+1
            right=len(nums)-1
            while left<right:
                if target>nums[left]+nums[right]:
                    left+=1
                elif target<nums[left]+nums[right]:
                    right-=1 
                else:
                    res.append([nums[i],nums[left],nums[right]])
                    while left<right and nums[left]==nums[left+1]:
                        left+=1
                    while left<right and nums[right]==nums[right-1]:
                        right-=1
                    left+=1
                    right-=1
        return res

#252. Meeting Rooms
#Time=O(n*logn) sorting ,O(n) for go through the array and determine if there is any overlap,space=O(1)
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        
        intervals.sort(key=lambda x: x[0])
    
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return False

        return True

#253. Meeting Rooms II
#Time=O(n*logn) sorting ,O(n) for go through the program, space=O(1)
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        start=[]
        end=[]
        for interval in intervals:
            start.append(interval[0])
            end.append(interval[1])
        start.sort()
        end.sort()
        s,e=0,0
        rooms,availability=0,0
        while s<len(start):
            if start[s]<end[e]:
                if availability==0:
                    rooms+=1
                else:
                    availability-=1
                s+=1
            else:
                availability+=1
                e+=1
        return rooms
    
#####################################################################################################
#string
#8. String to Integer (atoi)
#Time=O(n),spcae=O(1)
class Solution:
    def myAtoi(self, str: str) -> int:
        if not str:return 0
        n=len(str)
        base=0
        sign=1
        i=0
        while i<n and str[i]==' ':
            i+=1
        if i< n and (str[i]=='+' or str[i]=='-'):
            sign=1 if str[i]=='+' else -1
            i+=1
        while i<n and str[i]>='0' and str[i]<='9':
            if base > int((2**31-1)/10) or (base == int((2**31-1)/10) and str[i]>'7'):
                return 2**31-1 if sign==1 else -2**31
            base=10*base+ord(str[i])-ord('0')
            i+=1
        return base*sign

#20. Valid Parentheses
#Time=O(n), space=O(n)
class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        dict={')':'(','}':'{',']':'['}
        for i in s:
            if i in {'(','{','['}:
                stack.append(i)
            else:
                if not stack or dict[i] != stack[-1]:
                    return False
                stack.pop()
        return not stack
    
#32. Longest Valid Parentheses
#Time=O(n), space=O(n)
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        res=0
        stack=[-1]
        for i in range(len(s)):
            if s[i]=='(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    res=max(res,i-stack[-1])
        return res
    



