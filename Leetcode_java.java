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
