"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st


def app():
    """This funciton creates the home page"""
    # Add ballons animation
    st.balloons()

    # Add title to the page
    st.title('About Me')

    # Create two columns
    img_col, content_col = st.columns([1, 1])

    with img_col:
        # Add your photo here
        st.image("./images/my_photo.png")
        pass

    with content_col:
        # cust
        st.markdown("""<br><br><br>""", unsafe_allow_html=True)
        # Add your name 
        st.markdown('''### Name: [Shishir Shekhar]()''')
        # Add your email
        st.markdown('''### Email: [ShishirShekhar](mailto:sspdav02@gmail.com?subject=Iris%20Species%20Classifier&body=Hi%2C%20I%20saw%20your%20Iris%20Species%20Classifier.%0AI%20wanted%20to%20share%20my%20feedback%20with%20you.%0A%0AWrite%20your%20feedback%20here....)''')
        # Add your github
        st.markdown('''### GitHub: [ShishirShekhar](https://github.com/ShishirShekhar/)''')
        # Add your linkedin
        st.markdown('''### Linkedin: [Shishir-Shekhar](https://www.linkedin.com/in/shishir-shekhar/)''')