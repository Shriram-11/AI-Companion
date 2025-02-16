# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution(object):
    def longestZigZag(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        
        The function calculates the longest ZigZag path in a binary tree.
        A ZigZag path is defined as alternating between the left and right child nodes.
        We start from the root node and consider both directions (left and right), 
        recursively tracking the ZigZag path lengths, and keeping track of the maximum length found.
        """
        
        # Initialize the result variable to store the maximum ZigZag path length
        self.res = 0

        def zigzag(node, left, height):
            """
            Helper function to recursively explore the tree and calculate the longest ZigZag path.
            
            :param node: The current node being explored.
            :param left: A boolean value indicating the current direction.
                         True indicates the previous move was to the left, 
                         False indicates it was to the right.
            :param height: The length of the current ZigZag path.
            """
            
            # Base case: If the node is None, stop the recursion
            if not node:
                return
            
            # Update the result with the maximum ZigZag path length encountered so far
            self.res = max(self.res, height)
            
            # If we are currently moving left, the next step should move right, and vice versa
            if left:
                # Move to the left child, continuing the zigzag path
                zigzag(node.left, False, height + 1)  # Now move right after moving left
                # Start a new zigzag path from the right child
                zigzag(node.right, True, 1)  # Start moving left from the right child
            else:
                # Move to the right child, continuing the zigzag path
                zigzag(node.right, True, height + 1)  # Now move left after moving right
                # Start a new zigzag path from the left child
                zigzag(node.left, False, 1)  # Start moving right from the left child

        # Start the zigzag path from the root node, both to the left and right
        zigzag(root, True, 0)  # Start zigzag to the left
        zigzag(root, False, 0) # Start zigzag to the right
        
        # Return the maximum ZigZag path length found
        return self.res
