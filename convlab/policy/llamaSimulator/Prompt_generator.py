import json
import pprint
import re


def createLanguagePromptFromGoal(goal):

    prompt = f"""


    Pretend you are a USER who is looking for some services, and I am the customer service AGENT.
    Stay in character for every response you give me. Our conversation is based on the following goal delimited by triple backticks.
    The goal sets the context for the conversation. Stick to the given goal. 
    You should NOT generate all the information in the goal at once.
    You should generate short, precise, and informative response (less than 50 tokens), corresponding to only one or two items in the goal.
    It's crucial not to conclude the conversation until you've inquired about all the items in the goal, and in the specified order.
    Do not end the conversation if there are services that haven't been discussed yet.
    Only ask questions related to the services and attributes explicitly written in the goal.
    When you believe you have successfully gathered information about all the services and their attributes from the goal, you can conclude the conversation by expressing gratitude.
   
    
    The goal is:
    ```{describe_goal(goal)}```

    """
    return prompt


def describe_goal(goal):
    # Extract the last service domain key for later reference
    last_key = goal.domains[-1]

    # Initialize the message with the list of services the user wants to interact with
    msg = "Your goal consists of the following services: " + \
        str(goal.domains) + "\n"
    msg += "First, "

    # Loop through each service domain in the goal
    for key in goal.domain_goals.keys():
        # Describe the current service domain the user wants to interact with
        msg += "you have to ask for a (" + str(key) + "). You have to"

        # Check the goals associated with the current service domain
        for item in goal.domain_goals[key].keys():
            if item == 'inform':
                # If 'inform' is a goal, list the details the user needs to provide
                msg += " inform all the following details about the " + \
                    str(key) + ":\n"
                i = 1
                book_flag = False

                # Loop through each detail to be informed, except for booking details
                for x in goal.domain_goals[key]['inform']:
                    if 'book' not in x:
                        msg += str(i) + ") " + x + " = " + \
                            goal.domain_goals[key]['inform'][x] + "\n "
                        i += 1
                    else:
                        book_flag = True

                # If booking details are required, specify them separately
                if book_flag:
                    msg += "Also, ask the agent to book the " + \
                        str(key) + " for you. Specify the following booking details:\n"
                    for x in goal.domain_goals[key]['inform']:
                        if 'book' in x:
                            msg += x + " = " + \
                                goal.domain_goals[key]['inform'][x] + "\n "
                            book_flag = False

            if item == 'request':
                # If 'request' is a goal, list the questions the user needs to ask after agent's recommendation
                msg += "Once the agent recommends a " + \
                    str(key) + " for you, you have to request the values of the following(s):\n"
                i = 1
                for x in goal.domain_goals[key]['request']:
                    #msg += str(i) + ") " + str(key) + " " + x + "? \n"
                    msg += str(i) + ") " + x + "? \n"
                    i += 1

        # If it's not the last service domain, indicate that the user should move to the next one
        if key != last_key:
            msg += "\n\nThen, you move on to the next service, "

    return msg
