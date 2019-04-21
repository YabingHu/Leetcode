//Dynamic Programming:
//70. Climbing Stairs
// Time and Space O(n) for all methods


//Dvivde and Conquer
//169. Majority Element
//Both time and space are O(n)
class Solution {
    public int majorityElement(int[] nums) {
        Map<Integer,Integer> dict=new HashMap<Integer,Integer>();
        for (int i=0;i<nums.length;i++){
            if(dict.containsKey(nums[i])){
                dict.put(nums[i], dict.get(nums[i])+1);
                if(dict.get(nums[i])>(nums.length)/2) return nums[i];
            }
            else dict.put(nums[i],1);
            
        }
        return nums[0];//if only contains one element
    }
}

//Linked List


//Graph
//133. Clone Graph
/*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> neighbors;

    public Node() {}

    public Node(int _val,List<Node> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/
//DFS solution, time and space O(n)
class Solution {
    public Node cloneGraph(Node node) {
        HashMap<Node,Node> map=new HashMap<>();
        return helper(node,map);
    }
    public Node helper(Node node,HashMap<Node,Node> map){
        if(node==null) return null;
        if(map.containsKey(node)) return map.get(node);
        Node dup=new Node(node.val,new ArrayList<Node>());
        map.put(node,dup);
        for (Node neighbor:node.neighbors){
            Node clone=helper(neighbor,map);
            dup.neighbors.add(clone);
        }
        return dup;
    }
}

//BFS solution, time and space O(n)
class Solution {
    public Node cloneGraph(Node node) {
        if(node==null) return null;
        HashMap<Node,Node> map=new HashMap<>();
        Queue<Node> queue = new LinkedList<>();
        queue.add(node);
        Node dup=new Node(node.val,new ArrayList<Node>());
        map.put(node,dup);
        while(!queue.isEmpty()){
            Node t=queue.poll();
            for(Node neighbor : t.neighbors){
                 if(!map.containsKey(neighbor)){
                     map.put(neighbor,new Node(neighbor.val,new ArrayList<Node>()));
                     queue.add(neighbor);
                 }
                map.get(t).neighbors.add(map.get(neighbor));
            }
        }
        return dup;
    }
}

//138. Copy List with Random Pointer
//Time:O(n), space:O(n)
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Map<Node, Node> map = new HashMap<Node, Node>();
        Node node = head;
        // loop 1. copy all the nodes
        while (node != null) {
            map.put(node, new Node(node.val));
            node = node.next;
        }
         // loop 2. assign next and random pointers
         node = head;
         while (node != null) {
             map.get(node).next = map.get(node.next);
             map.get(node).random = map.get(node.random);
             node = node.next;
         }
         return map.get(head);
    }
}




//200. Number of Islands
//Time and space O(m*n)
class Solution {
    public int numIslands(char[][] grid) {
        if(grid.length==0) return 0;
        int m=grid.length, n=grid[0].length,res=0;
        int[][] visited=new int[m][n];
        for (int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]=='1' && visited[i][j]==0){
                    helper(grid,visited,i,j);
                    res++;
                }
            }
        } 
    return res;
    }
    
    public void helper(char[][] grid,int[][] visited,int x,int y){
        if(x<0 ||x>= grid.length||y<0||y>=grid[0].length||grid[x][y]=='0'||visited[x][y]==1) return;
        visited[x][y]=1;
        helper(grid,visited,x-1,y);
        helper(grid,visited,x+1,y);
        helper(grid,visited,x,y-1);
        helper(grid,visited,x,y+1);
    }
}

//547. Friend Circles
//Time=O(n^2), space=O(n)
class Solution {
    public int findCircleNum(int[][] M) {
        if(M.length==0) return 0;
        int m=M.length;
        int[] visited=new int[m];
        int res=0;
        for (int i=0;i<m;i++){
            if (visited[i]==1) continue;
            helper(M,i,m,visited);
            res++;
        }
        return res;
    }
    public void helper(int[][] M,int i ,int m,int[] visited){
            if(visited[i]==1) return;
            visited[i]=1;
            for(int j=0;j<m;j++){
                if(M[i][j]==1 && visited[j]==0)
                helper(M,j,m,visited);
            }
    }
}



//Binary Search
//35. Search Insert Position
//Time=O(logn),sapce=O(1)
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left=0;
        int right=nums.length-1;
        while (left<=right){
            int mid=left+(right-left)/2;
            if(target>nums[mid]) left=mid+1;
            else right=mid-1;
        }
        return right+1;
    }
}

//34. Find First and Last Position of Element in Sorted Array
//Time=O(logn),sapce=O(1)
//Two binary search solution
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] res=new int[2];
        res[0]=-1;
        res[1]=-1;
        if(nums.length==0) return res;
        int left=0;
        int right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]<target) left=mid+1;
            else right=mid-1;
        }
        if(right+1==nums.length) return res;
        if(nums[right+1]!=target) return res;
        res[0]=right+1;
        right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]<=target) left=mid+1;
            else right=mid-1;
        }
        res[1]=right;
        return res;
    }
}

//704. Binary Search
//Time=O(logn),sapce=O(1)
class Solution {
    public int search(int[] nums, int target) {
        int left=0;
        int right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if (nums[mid]==target) return mid;
            else if (nums[mid]>target) right=mid-1;
            else left=mid+1;
        }
        return -1;
    }
}

//33. Search in Rotated Sorted Array
//Time=O(logn),sapce=O(1)
class Solution {
    public int search(int[] nums, int target) {
        int left=0;
        int right=nums.length-1;
        while(left<=right){
            int mid=left+(right-left)/2;
            if(nums[mid]==target) return mid;
            else if (nums[mid]<nums[right]){
                if(nums[mid]<target && nums[right]>=target) left=mid+1;
                else right=mid-1;
            }
            else{
                if(nums[mid]>target&& nums[left]<=target) right=mid-1;
                else left=mid+1;
            }
        }
        return -1;
    }
}
